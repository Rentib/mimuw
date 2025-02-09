use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

type Task = Box<dyn FnOnce() + Send>;

// You can define new types (e.g., structs) if you need.
// However, they shall not be public (i.e., do not use the `pub` keyword).

/// The thread pool.
pub struct Threadpool {
    // Add here any fields you need.
    // We suggest storing handles of the worker threads, submitted tasks,
    // and information whether the pool is running or is shutting down.
    thrds: Vec<JoinHandle<()>>,
    tasks: Arc<(Mutex<Vec<Task>>, Condvar)>,
    running: Arc<AtomicBool>,
}

impl Threadpool {
    /// Create new thread pool with `workers_count` workers.
    pub fn new(workers_count: usize) -> Self {
        let mut thrds: Vec<JoinHandle<()>> = Vec::new();
        let tasks = Arc::new((Mutex::new(Vec::new()), Condvar::new()));
        let running = Arc::new(AtomicBool::new(true));

        for _ in 0..workers_count {
            let tasks = Arc::clone(&tasks);
            let running = Arc::clone(&running);
            let thrd = std::thread::spawn(|| {
                Self::worker_loop(tasks, running);
            });

            thrds.push(thrd);
        }

        Self {
            thrds,
            tasks,
            running,
        }
    }

    /// Submit a new task.
    pub fn submit(&self, task: Task) {
        let (mtx, cnd) = self.tasks.as_ref();
        let mut tasks = mtx.lock().unwrap();
        tasks.push(task);
        cnd.notify_one();
    }

    // We suggest extracting the implementation of the worker to an associated
    // function, like this one (however, it is not a part of the public
    // interface, so you can delete it if you implement it differently):
    fn worker_loop(tasks: Arc<(Mutex<Vec<Task>>, Condvar)>, running: Arc<AtomicBool>) {
        loop {
            let (mtx, cnd) = tasks.as_ref();
            let mut tasks = mtx.lock().unwrap();
            while tasks.is_empty() && running.load(Ordering::Relaxed) {
                tasks = cnd.wait(tasks).unwrap();
            }

            if tasks.is_empty() && !running.load(Ordering::Relaxed) {
                break;
            }

            let task = tasks.pop().unwrap();
            drop(tasks);
            task();
        }
    }
}

impl Drop for Threadpool {
    /// Gracefully end the thread pool.
    ///
    /// It waits until all submitted tasks are executed,
    /// and until all threads are joined.
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        let (_, cnd) = self.tasks.as_ref();
        cnd.notify_all();
        for thrd in self.thrds.drain(..) {
            thrd.join().unwrap();
        }
    }
}
