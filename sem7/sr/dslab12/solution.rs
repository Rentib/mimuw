use module_system::{Handler, ModuleRef, System};
use std::collections::{HashSet, VecDeque}; // Who thought of not implementing a normal Queue???

/// Marker trait indicating that a broadcast implementation provides
/// guarantees specified in the assignment description.
pub(crate) trait ReliableBroadcast<const N: usize> {}

#[async_trait::async_trait]
pub(crate) trait ReliableBroadcastRef<const N: usize>: Send + Sync + 'static {
    async fn send(&self, msg: Operation);
}

#[async_trait::async_trait]
impl<T, const N: usize> ReliableBroadcastRef<N> for ModuleRef<T>
where
    T: ReliableBroadcast<N> + Handler<Operation> + Send,
{
    async fn send(&self, msg: Operation) {
        self.send(msg).await;
    }
}

/// Marker trait indicating that a client implementation
/// follows specification from the assignment description.
pub(crate) trait EditorClient {}

#[async_trait::async_trait]
pub(crate) trait ClientRef: Send + Sync + 'static {
    async fn send(&self, msg: Edit);
}

#[async_trait::async_trait]
impl<T> ClientRef for ModuleRef<T>
where
    T: EditorClient + Handler<Edit> + Send,
{
    async fn send(&self, msg: Edit) {
        self.send(msg).await;
    }
}

/// Actions (edits) which can be applied to a text.
#[derive(Clone)]
#[cfg_attr(test, derive(PartialEq, Debug))]
pub(crate) enum Action {
    /// Insert the character at the position.
    Insert { idx: usize, ch: char },
    /// Delete a character at the position.
    Delete { idx: usize },
    /// A _do nothing_ operation. `Nop` cannot be issued by a client.
    /// `Nop` can only be issued by a process or result from a transformation.
    Nop,
}

impl Action {
    /// Apply the action to the text.
    pub(crate) fn apply_to(&self, text: &mut String) {
        match self {
            Action::Insert { idx, ch } => {
                text.insert(*idx, *ch);
            }
            Action::Delete { idx } => {
                text.remove(*idx);
            }
            Action::Nop => {
                // Do nothing.
            }
        }
    }
}

/// Client's request to edit the text.
#[derive(Clone)]
pub(crate) struct EditRequest {
    /// Total number of operations a client has applied to its text so far.
    pub(crate) num_applied: usize,
    /// Action (edit) to be applied to a text.
    pub(crate) action: Action,
}

/// Response to a client with action (edit) it should apply to its text.
#[derive(Clone)]
pub(crate) struct Edit {
    pub(crate) action: Action,
}

#[derive(Clone)]
pub(crate) struct Operation {
    /// Rank of a process which issued this operation.
    pub(crate) process_rank: usize,
    /// Action (edit) to be applied to a text.
    pub(crate) action: Action,
}

impl Operation {
    // Add any methods you need.
    fn transform(op1: Operation, op2: &Operation) -> Operation {
        let (r1, r2) = (op1.process_rank, op2.process_rank);
        Operation {
            process_rank: r1,
            action: match (op1.action.clone(), op2.action.clone()) {
                (Action::Insert { idx: p1, ch: c1 }, Action::Insert { idx: p2, ch: _c2 }) => {
                    Action::Insert {
                        idx: if p1 < p2 || (p1 == p2 && r1 < r2) {
                            p1
                        } else {
                            p1 + 1
                        },
                        ch: c1,
                    }
                }
                (Action::Delete { idx: p1 }, Action::Delete { idx: p2 }) if p1 < p2 => {
                    Action::Delete { idx: p1 }
                }
                (Action::Delete { idx: p1 }, Action::Delete { idx: p2 }) if p1 == p2 => Action::Nop,
                (Action::Delete { idx: p1 }, Action::Delete { idx: _p2 }) => {
                    Action::Delete { idx: p1 - 1 }
                }
                (Action::Insert { idx: p1, ch: c1 }, Action::Delete { idx: p2 }) => {
                    Action::Insert {
                        idx: if p1 <= p2 { p1 } else { p1 - 1 },
                        ch: c1,
                    }
                }
                (Action::Delete { idx: p1 }, Action::Insert { idx: p2, ch: _c2 }) => {
                    Action::Delete {
                        idx: if p1 < p2 { p1 } else { p1 + 1 },
                    }
                }
                (action1, _action2) => action1, // Nop x Any = Nop, Any x Nop = Any
            },
        }
    }
}

/// Process of the system.
pub(crate) struct Process<const N: usize> {
    /// Rank of the process.
    rank: usize,
    /// Reference to the broadcast module.
    broadcast: Box<dyn ReliableBroadcastRef<N>>,
    /// Reference to the process's client.
    client: Box<dyn ClientRef>,
    /// Buffer for requests waiting to be processed.
    requests: VecDeque<EditRequest>,
    /// Operations done by this process.
    operations: Vec<Operation>,
    /// Processes already handled in the current round.
    curr_round: HashSet<usize>,
    /// Operations that have been received in the current round but need to be processed in the next one.
    next_round: VecDeque<Operation>,
}

impl<const N: usize> Process<N> {
    pub(crate) async fn new(
        system: &mut System,
        rank: usize,
        broadcast: Box<dyn ReliableBroadcastRef<N>>,
        client: Box<dyn ClientRef>,
    ) -> ModuleRef<Self> {
        system
            .register_module(Self {
                rank,
                broadcast,
                client,
                // Add any fields you need.
                operations: Vec::new(),
                requests: VecDeque::new(),
                curr_round: HashSet::new(),
                next_round: VecDeque::new(),
            })
            .await
    }

    // Add any methods you need.
    async fn process_request(&mut self, request: EditRequest) {
        // Apply the operation to the text combined with all the previously applied operations
        let op = Operation {
            process_rank: self.rank,
            action: self.operations[request.num_applied..]
                .iter()
                .fold(
                    Operation {
                        process_rank: N + 1,
                        action: request.action,
                    },
                    |op1, op2| -> Operation { Operation::transform(op1, op2) },
                )
                .action,
        };

        self.operations.push(op.clone());
        self.curr_round.insert(self.rank);

        self.broadcast.send(op.clone()).await;
        self.client.send(Edit { action: op.action }).await;
    }
}

#[async_trait::async_trait]
impl<const N: usize> Handler<Operation> for Process<N> {
    async fn handle(&mut self, _self_ref: &ModuleRef<Self>, msg: Operation) {
        // If we have already handled the message of this process in this round, save it for the next one
        if self.curr_round.contains(&msg.process_rank) {
            self.next_round.push_back(msg);
            return;
        }

        if self.curr_round.is_empty() {
            self.process_request(EditRequest {
                num_applied: self.operations.len(),
                action: Action::Nop,
            })
            .await;
        }

        let op = self.operations[self.operations.len() - self.curr_round.len()..]
            .iter()
            .fold(msg.clone(), |op1, op2| -> Operation {
                Operation::transform(op1, op2)
            });

        self.operations.push(op.clone());
        self.curr_round.insert(msg.process_rank);

        let action = op.action.clone();
        self.client.send(Edit { action }).await;

        if self.curr_round.len() < N {
            return;
        }

        // We have finished the round, so we can process the next buffered request
        self.curr_round.clear();

        // TODO: check if this should be before or after the following while
        if let Some(req) = self.requests.pop_front() {
            self.process_request(req).await
        }

        while let Some(msg) = self.next_round.pop_front() {
            self.handle(_self_ref, msg).await;
        }
    }
}

#[async_trait::async_trait]
impl<const N: usize> Handler<EditRequest> for Process<N> {
    async fn handle(&mut self, _self_ref: &ModuleRef<Self>, request: EditRequest) {
        if self.curr_round.is_empty() {
            // If no responses are pending, process the request immediately
            self.process_request(request).await;
        } else {
            // Otherwise, buffer the request
            self.requests.push_back(request);
        }
    }
}
