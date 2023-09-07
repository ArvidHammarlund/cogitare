use burn::{lr_scheduler::LRScheduler, LearningRate};

pub struct PeriodicLr {
    step: usize,
    start_base: usize,
}

impl LRScheduler for PeriodicLr {
    type Record = usize;

    fn step(&mut self) -> LearningRate {
        self.step += 1;
        let floor = f64::log10(self.step as f64).floor() as i32;
        100f64.powi(-(floor + self.start_base as i32))
    }

    fn to_record(&self) -> Self::Record {
        self.step as usize
    }

    fn load_record(mut self, record: Self::Record) -> Self {
        self.step = record as usize;
        self
    }
}

impl PeriodicLr {
    pub fn new(start_base: usize) -> Self {
        Self {
            step: 0,
            start_base,
        }
    }
}
