use std::collections::HashSet;
use std::time::Duration;

use uuid::Uuid;

use module_system::{Handler, ModuleRef, System, TimerHandle};

/// State of a Raft process.
/// It shall be kept in stable storage, and updated before replying to messages.
#[derive(Default, Clone, Copy)]
pub(crate) struct ProcessState {
    /// Number of the current term. `0` at boot.
    pub(crate) current_term: u64,
    /// Identifier of a process which has received this process' vote.
    /// `None if this process has not voted in this term.
    voted_for: Option<Uuid>,
    /// Identifier of a process which is thought to be the leader.
    leader_id: Option<Uuid>,
}

/// Configuration of a Raft process.
#[derive(Copy, Clone)]
pub(crate) struct ProcessConfig {
    /// UUID of the process.
    pub(crate) self_id: Uuid,
    /// Timeout used by this process.
    /// In contrast to real-world applications, in this assignment,
    /// the timeout shall not be randomized!
    pub(crate) election_timeout: Duration,
    /// Number of processes in the system.
    /// In the assignment, there is a constant number of processes.
    pub(crate) processes_count: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct RaftMessage {
    pub(crate) header: RaftMessageHeader,
    pub(crate) content: RaftMessageContent,
}

#[derive(Clone)]
struct Timeout;

struct Init;

/// Message disabling a process. Used for testing to simulate failures.
pub(crate) struct Disable;

#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub(crate) struct RaftMessageHeader {
    /// Term of the process which issues the message.
    pub(crate) term: u64,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum RaftMessageContent {
    Heartbeat {
        /// Id of the process issuing the message, which claims to be the leader.
        leader_id: Uuid,
    },
    HeartbeatResponse,
    RequestVote {
        /// Id of the process issuing the message, which is a candidate.
        candidate_id: Uuid,
    },
    RequestVoteResponse {
        /// Whether the vote was granted or not.
        granted: bool,
        /// Id of the process issuing the message, which grants (or not) the vote.
        source: Uuid,
    },
}

pub(crate) trait StableStorage: Send {
    fn put(&mut self, state: &ProcessState);

    fn get(&self) -> Option<ProcessState>;
}

#[async_trait::async_trait]
pub(crate) trait Sender: Send + Sync {
    async fn send(&self, target: &Uuid, msg: RaftMessage);

    async fn broadcast(&self, msg: RaftMessage);
}

/// Process of Raft.
pub struct Raft {
    state: ProcessState,
    config: ProcessConfig,
    storage: Box<dyn StableStorage>,
    sender: Box<dyn Sender>,
    process_type: ProcessType,
    timer_handle: Option<TimerHandle>,
    enabled: bool,
    self_ref: Option<ModuleRef<Self>>,
}

impl Raft {
    pub(crate) async fn new(
        system: &mut System,
        config: ProcessConfig,
        storage: Box<dyn StableStorage>,
        sender: Box<dyn Sender>,
    ) -> ModuleRef<Self> {
        let state = storage.get().unwrap_or_default();

        let self_ref = system
            .register_module(Self {
                state,
                config,
                storage,
                sender,
                process_type: Default::default(),
                timer_handle: None,
                enabled: true,
                self_ref: None,
            })
            .await;
        self_ref.send(Init).await;
        self_ref
    }

    async fn reset_timer(&mut self, interval: Duration) {
        if let Some(handle) = self.timer_handle.take() {
            handle.stop().await;
        }
        self.timer_handle = Some(
            self.self_ref
                .as_ref()
                .unwrap()
                .request_tick(Timeout, interval)
                .await,
        );
    }

    /// Set the process's term to the higher number.
    fn update_term(&mut self, new_term: u64) {
        assert!(self.state.current_term < new_term);
        self.state.current_term = new_term;
        self.state.voted_for = None;
        self.state.leader_id = None;
        // No reliable state update called here, must be called separately.
    }

    /// Reliably save the state.
    fn update_state(&mut self) {
        self.storage.put(&self.state);
    }

    /// Handle the received heartbeat.
    async fn heartbeat(&mut self, leader_id: Uuid, leader_term: u64) {
        if leader_term >= self.state.current_term {
            self.state.leader_id = Some(leader_id);
            self.update_state();

            // Update the volatile state:
            match &mut self.process_type {
                ProcessType::Follower => {
                    self.reset_timer(self.config.election_timeout).await;
                }
                ProcessType::Candidate { .. } => {
                    self.process_type = ProcessType::Follower;
                    self.reset_timer(self.config.election_timeout).await;
                }
                ProcessType::Leader => {
                    log::debug!("Ignore, heartbeat from self")
                }
            };
        }

        // Response is always sent, so information about the term is disseminated:
        self.sender
            .send(
                &leader_id,
                RaftMessage {
                    header: RaftMessageHeader {
                        term: self.state.current_term,
                    },
                    content: RaftMessageContent::HeartbeatResponse,
                },
            )
            .await;
    }

    /// Common message processing.
    fn msg_received(&mut self, msg: &RaftMessage) {
        if msg.header.term > self.state.current_term {
            self.update_term(msg.header.term);
            self.process_type = ProcessType::Follower;
        }
    }

    /// Broadcast heartbeat.
    async fn broadcast_heartbeat(&mut self) {
        self.sender
            .broadcast(RaftMessage {
                header: RaftMessageHeader {
                    term: self.state.current_term,
                },
                content: RaftMessageContent::Heartbeat {
                    leader_id: self.config.self_id,
                },
            })
            .await;
    }

    fn vote(&mut self, candidate_id: Uuid) -> bool {
        self.state.voted_for = Some(candidate_id);
        self.update_state();
        true
    }

    async fn candidate(&mut self) {
        self.process_type = ProcessType::Candidate {
            votes_received: HashSet::new(),
        };
        self.update_term(self.state.current_term + 1);
        self.update_state();

        let msg = RaftMessage {
            header: RaftMessageHeader {
                term: self.state.current_term,
            },
            content: RaftMessageContent::RequestVote {
                candidate_id: self.config.self_id,
            },
        };

        self.sender.broadcast(msg).await;
    }
}

#[async_trait::async_trait]
impl Handler<Init> for Raft {
    async fn handle(&mut self, self_ref: &ModuleRef<Self>, _msg: Init) {
        if self.enabled {
            self.self_ref = Some(self_ref.clone());
            self.reset_timer(self.config.election_timeout).await;
        }
    }
}

/// Handle timer timeout.
#[async_trait::async_trait]
impl Handler<Timeout> for Raft {
    async fn handle(&mut self, _self_ref: &ModuleRef<Self>, _: Timeout) {
        if self.enabled {
            match &mut self.process_type {
                ProcessType::Follower {} => {
                    if self.state.voted_for == self.state.leader_id {
                        self.candidate().await;
                    }
                }
                ProcessType::Candidate { .. } => self.candidate().await,
                ProcessType::Leader => self.broadcast_heartbeat().await,
            }
        }
    }
}

#[async_trait::async_trait]
impl Handler<RaftMessage> for Raft {
    async fn handle(&mut self, _self_ref: &ModuleRef<Self>, msg: RaftMessage) {
        if self.enabled {
            // Update common state:
            self.msg_received(&msg);

            // TODO message specific processing. Heartbeat is given as an example:
            match (&mut self.process_type, msg.content) {
                (_, RaftMessageContent::Heartbeat { leader_id }) => {
                    self.heartbeat(leader_id, msg.header.term).await;
                }
                (_, RaftMessageContent::RequestVote { candidate_id }) => {
                    if msg.header.term < self.state.current_term {
                        return;
                    }
                    let mut granted = false;
                    match self.process_type {
                        ProcessType::Candidate { .. } => {
                            if candidate_id != self.config.self_id {
                                self.process_type = ProcessType::Follower;
                            }
                            granted = self.vote(candidate_id);
                        }
                        ProcessType::Follower => {
                            if self.state.voted_for == self.state.leader_id {
                                granted = self.vote(candidate_id);
                            }
                        }
                        ProcessType::Leader => {}
                    }

                    let msg = RaftMessage {
                        header: RaftMessageHeader {
                            term: self.state.current_term,
                        },
                        content: RaftMessageContent::RequestVoteResponse {
                            granted,
                            source: self.config.self_id,
                        },
                    };

                    self.sender.send(&candidate_id, msg).await;
                }
                (
                    ProcessType::Candidate { votes_received },
                    RaftMessageContent::RequestVoteResponse { granted, source },
                ) => {
                    if granted {
                        votes_received.insert(source);
                        if 2 * votes_received.len() > self.config.processes_count {
                            self.process_type = ProcessType::Leader;
                            self.reset_timer(self.config.election_timeout.checked_div(10).unwrap())
                                .await;
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

#[async_trait::async_trait]
impl Handler<Disable> for Raft {
    async fn handle(&mut self, _self_ref: &ModuleRef<Self>, _: Disable) {
        self.enabled = false;
    }
}

/// State of a Raft process with a corresponding (volatile) information.
#[derive(Default)]
enum ProcessType {
    #[default]
    Follower,
    Candidate {
        votes_received: HashSet<Uuid>,
    },
    Leader,
}
