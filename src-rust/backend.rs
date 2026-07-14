use std::collections::VecDeque;
use std::process::Stdio;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;

use crate::protocol::{BackendEvent, EventFrame, Request, MAX_FRAME_BYTES};

const EVENT_CAPACITY: usize = 2048;
const COMMAND_CAPACITY: usize = 256;
const DIAGNOSTIC_BYTES: usize = 64 * 1024;

pub struct Backend {
    commands: mpsc::Sender<Request>,
    pub events: Receiver<BackendEvent>,
    join: Option<thread::JoinHandle<()>>,
    child_pid: Arc<AtomicI32>,
}

async fn read_bounded_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
) -> Result<Option<String>, String> {
    let mut bytes = Vec::with_capacity(8192);
    let mut oversized = false;
    loop {
        let available = reader
            .fill_buf()
            .await
            .map_err(|error| format!("runtime read failed: {error}"))?;
        if available.is_empty() {
            if bytes.is_empty() && !oversized {
                return Ok(None);
            }
            break;
        }
        let take = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        if !oversized {
            if bytes.len().saturating_add(take) > MAX_FRAME_BYTES {
                oversized = true;
                bytes.clear();
            } else {
                bytes.extend_from_slice(&available[..take]);
            }
        }
        let found_newline = available[take - 1] == b'\n';
        reader.consume(take);
        if found_newline {
            break;
        }
    }
    if oversized {
        return Err("runtime frame exceeds 1 MiB".into());
    }
    if bytes.last() == Some(&b'\n') {
        bytes.pop();
        if bytes.last() == Some(&b'\r') {
            bytes.pop();
        }
    }
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|_| "runtime frame is not valid UTF-8".into())
}

#[cfg(unix)]
fn signal_process_group(pid: i32, signal: i32) {
    if pid > 0 {
        // The child is created as its own process group leader.
        unsafe {
            libc::kill(-pid, signal);
        }
    }
}

#[cfg(not(unix))]
fn signal_process_group(_pid: i32, _signal: i32) {}

impl Backend {
    pub fn start(python: &str, project_root: Option<&str>, debug: bool) -> Result<Self, String> {
        let (command_tx, mut command_rx) = mpsc::channel::<Request>(COMMAND_CAPACITY);
        let (event_tx, event_rx) = bounded::<BackendEvent>(EVENT_CAPACITY);
        let python = python.to_owned();
        let project_root = project_root.map(str::to_owned);
        let child_pid = Arc::new(AtomicI32::new(0));
        let runtime_child_pid = Arc::clone(&child_pid);
        let join = thread::Builder::new()
            .name("alphanus-runtime".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build();
                let Ok(runtime) = runtime else {
                    let _ = event_tx.send(BackendEvent::ProtocolError("failed to create Tokio runtime".into()));
                    return;
                };
                runtime.block_on(async move {
                    let mut command = Command::new(&python);
                    command.arg("-m").arg("alphanus_cli").arg("_runtime");
                    if let Some(root) = project_root {
                        command.arg("--project-root").arg(root);
                    }
                    if debug {
                        command.arg("--debug");
                    }
                    command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
                    #[cfg(unix)]
                    command.process_group(0);
                    let mut child = match command.spawn() {
                        Ok(child) => child,
                        Err(error) => {
                            let _ = event_tx.send(BackendEvent::ProtocolError(format!("failed to start runtime: {error}")));
                            return;
                        }
                    };
                    runtime_child_pid.store(child.id().unwrap_or(0) as i32, Ordering::Release);
                    let Some(mut stdin) = child.stdin.take() else {
                        let _ = event_tx.send(BackendEvent::ProtocolError("runtime stdin unavailable".into()));
                        return;
                    };
                    let Some(stdout) = child.stdout.take() else {
                        let _ = event_tx.send(BackendEvent::ProtocolError("runtime stdout unavailable".into()));
                        return;
                    };
                    let Some(stderr) = child.stderr.take() else {
                        let _ = event_tx.send(BackendEvent::ProtocolError("runtime stderr unavailable".into()));
                        return;
                    };

                    let stderr_tx = event_tx.clone();
                    tokio::spawn(async move {
                        let mut lines = BufReader::new(stderr).lines();
                        let mut retained = VecDeque::<String>::new();
                        let mut retained_bytes = 0usize;
                        while let Ok(Some(line)) = lines.next_line().await {
                            retained_bytes += line.len();
                            retained.push_back(line.clone());
                            while retained_bytes > DIAGNOSTIC_BYTES {
                                if let Some(old) = retained.pop_front() {
                                    retained_bytes = retained_bytes.saturating_sub(old.len());
                                } else {
                                    break;
                                }
                            }
                            let clipped = if line.len() > 4096 { format!("{}…", &line[..4096]) } else { line };
                            let _ = stderr_tx.send(BackendEvent::Diagnostic(clipped));
                        }
                    });

                    let mut stdout = BufReader::new(stdout);
                    loop {
                        tokio::select! {
                            command = command_rx.recv() => {
                                let Some(command) = command else { break; };
                                match serde_json::to_vec(&command) {
                                    Ok(encoded) if encoded.len() <= MAX_FRAME_BYTES => {
                                        if stdin.write_all(&encoded).await.is_err()
                                            || stdin.write_all(b"\n").await.is_err()
                                            || stdin.flush().await.is_err()
                                        {
                                            break;
                                        }
                                    }
                                    Ok(_) => {
                                        let _ = event_tx.send(BackendEvent::ProtocolError("outbound runtime frame exceeds 1 MiB".into()));
                                    }
                                    Err(error) => {
                                        let _ = event_tx.send(BackendEvent::ProtocolError(format!("failed to encode request: {error}")));
                                    }
                                }
                            }
                            line = read_bounded_line(&mut stdout) => {
                                match line {
                                    Ok(Some(line)) => match EventFrame::decode(&line) {
                                        Ok(frame) => {
                                            if event_tx.send(BackendEvent::Frame(frame)).is_err() { break; }
                                        }
                                        Err(error) => {
                                            let _ = event_tx.send(BackendEvent::ProtocolError(error));
                                        }
                                    },
                                    Ok(None) => break,
                                    Err(error) => {
                                        let _ = event_tx.send(BackendEvent::ProtocolError(error));
                                    }
                                }
                            }
                        }
                    }
                    let pid = child.id().unwrap_or(0) as i32;
                    #[cfg(unix)]
                    signal_process_group(pid, libc::SIGTERM);
                    #[cfg(not(unix))]
                    let _ = child.start_kill();
                    let status = match tokio::time::timeout(Duration::from_secs(2), child.wait()).await {
                        Ok(result) => result.ok(),
                        Err(_) => {
                            #[cfg(unix)]
                            signal_process_group(pid, libc::SIGKILL);
                            #[cfg(not(unix))]
                            let _ = child.start_kill();
                            child.wait().await.ok()
                        }
                    };
                    runtime_child_pid.store(0, Ordering::Release);
                    let _ = event_tx.send(BackendEvent::Exited(status.and_then(|value| value.code())));
                });
            })
            .map_err(|error| format!("failed to create runtime thread: {error}"))?;
        let backend = Self {
            commands: command_tx,
            events: event_rx,
            join: Some(join),
            child_pid,
        };
        backend.send(Request::hello())?;
        Ok(backend)
    }

    pub fn send(&self, request: Request) -> Result<(), String> {
        self.commands
            .try_send(request)
            .map_err(|error| format!("runtime command queue unavailable: {error}"))
    }

    pub fn shutdown(&mut self) {
        let _ = self.send(Request::new("shutdown", serde_json::json!({})));
        let (replacement, _receiver) = mpsc::channel(1);
        self.commands = replacement;
        if let Some(join) = self.join.take() {
            let deadline = Instant::now() + Duration::from_secs(7);
            while !join.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(25));
            }
            if !join.is_finished() {
                let pid = self.child_pid.load(Ordering::Acquire);
                #[cfg(unix)]
                signal_process_group(pid, libc::SIGKILL);
            }
            let _ = join.join();
        }
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        self.shutdown();
    }
}
