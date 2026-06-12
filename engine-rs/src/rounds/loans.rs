//! Corporation loans (1867's InterestOnLoans economy, Ruby game.rb:499-530
//! + step/automatic_loan.rb).
//!
//! A loan has a face value (1867: $50); TAKING one nets `value - interest`
//! ($45) from the bank, REPAYING costs the full face. Majors hold ≤5,
//! minors ≤2, the bank pool holds 72. `AutomaticLoan` steps (Track,
//! BuyTrain) take loans implicitly while a cost exceeds cash.
//!
//! TODO(1867-interest): interest snapshots at OR start + the LoanOperations
//! auto-step (pay interest → nationalize if unpayable → auto-repay) land
//! with the loan-operations seam; this module is the take/spend half the
//! OR1 frontier needs.

use crate::actions::GameError;
use crate::game::BaseGame;

impl BaseGame {
    /// Max loans the corp's class may hold (0 for titles without loans).
    pub(crate) fn corp_max_loans(&self, corp_idx: usize) -> u32 {
        self.title_def()
            .max_loans(self.corporations[corp_idx].corp_type)
    }

    /// Whether the corp can take one more loan (Ruby `can_take_loan?`).
    pub(crate) fn corp_can_take_loan(&self, corp_idx: usize) -> bool {
        self.title_def().loan_value() > 0
            && self.loans_remaining > 0
            && self.corporations[corp_idx].loans < self.corp_max_loans(corp_idx)
    }

    /// Cash still raisable through loans: takeable loans × the per-loan net
    /// (Ruby `buying_power(full:)`'s loan component).
    pub(crate) fn corp_loan_capacity_cash(&self, corp_idx: usize) -> i32 {
        let title = self.title_def();
        let value = title.loan_value();
        if value == 0 {
            return 0;
        }
        let per_loan = value - title.loan_interest_rate();
        let takeable = self
            .corp_max_loans(corp_idx)
            .saturating_sub(self.corporations[corp_idx].loans)
            .min(self.loans_remaining);
        takeable as i32 * per_loan
    }

    /// Ruby `buying_power(entity, full: true)`: cash + loan capacity.
    pub(crate) fn corp_buying_power_full(&self, corp_idx: usize) -> i32 {
        self.corporations[corp_idx].cash + self.corp_loan_capacity_cash(corp_idx)
    }

    /// Ruby `take_loan`: the bank pays `value - interest`, the loan moves to
    /// the corporation.
    pub(crate) fn take_loan(&mut self, corp_idx: usize) -> Result<(), GameError> {
        if !self.corp_can_take_loan(corp_idx) {
            return Err(GameError::new(format!(
                "Cannot take more than {} loans",
                self.corp_max_loans(corp_idx)
            )));
        }
        let title = self.title_def();
        let net = title.loan_value() - title.loan_interest_rate();
        self.corporations[corp_idx].cash += net;
        self.bank.cash -= net;
        self.corporations[corp_idx].loans += 1;
        self.loans_remaining -= 1;
        Ok(())
    }

    /// AutomaticLoan#try_take_loan: take loans until `cost` is covered (or
    /// capacity runs out — the caller's normal cash check then fails).
    pub(crate) fn auto_take_loans(&mut self, corp_idx: usize, cost: i32) {
        while self.corporations[corp_idx].cash < cost && self.corp_can_take_loan(corp_idx) {
            let _ = self.take_loan(corp_idx);
        }
    }

    /// The current train limit for THIS corp: the phase's per-class limit
    /// (1867 phases carry `{minor: N, major: M}`; 1830 a single number).
    pub(crate) fn corp_train_limit(&self, corp_idx: usize) -> usize {
        if self.corporations[corp_idx].corp_type == crate::title::CorpType::Minor {
            if let Some(pd) = self
                .title_def()
                .phases()
                .iter()
                .find(|p| p.name == self.phase.name)
            {
                if let Some(ml) = pd.minor_train_limit {
                    return ml as usize;
                }
            }
        }
        self.phase.train_limit as usize
    }
}
