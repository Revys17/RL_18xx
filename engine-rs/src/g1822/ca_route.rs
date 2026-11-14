//! 1822CA routes: the Route step's pullman and grain-train attachments
//! (CA Route#choosing? / #choices / #process_choose, "None" skips), the
//! grain train left out of `route_trains`, and the small mail contracts.

use crate::actions::GameError;
use crate::entities::Train;
use crate::game::BaseGame;

use super::ca::PRIVATE_SMALL_MAIL_CONTRACTS;
use super::err;
use super::operating::ruby_to_i;

/// Route `@choosing`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaChoosing {
    Pullman,
    Grain,
}

/// CA Route::CHOICE_SKIP.
pub const CHOICE_SKIP: &str = "None";

impl BaseGame {
    /// `grain_train?(train)`.
    pub(crate) fn g1822_ca_grain_train(t: &Train) -> bool {
        t.name == "G"
    }

    /// A train name without the grain attachment's suffixes:
    /// `attach_grain_train` appends "-G" after a P and "G" otherwise
    /// ("LP" -> "LP-G", "3" -> "3G"), and the name can outlive the
    /// attachment ([`Self::g1822_ca_route_setup`]): "LP-G" -> "LP", "LG" ->
    /// "L", "LP-GG" (a left-over name attached again) -> "LP". The grain
    /// train "G" itself and every other name are returned unchanged.
    pub(crate) fn g1822_ca_ungrained(name: &str) -> &str {
        let mut n = name;
        loop {
            if let Some(b) = n.strip_suffix("-G").filter(|b| b.ends_with('P')) {
                n = b;
            } else if let Some(b) = n.strip_suffix('G').filter(|b| !b.is_empty() && !b.ends_with('-')) {
                n = b;
            } else {
                return n;
            }
        }
    }

    /// A grain name left on a train that is not (any more) the Route step's
    /// grain-attached train ([`Self::g1822_ca_route_setup`]).
    pub(crate) fn g1822_ca_left_grain_name(&self, t: &Train) -> bool {
        Self::g1822_ca_ungrained(&t.name) != t.name && self.rnd().ca.grain_train.as_deref() != Some(t.id.as_str())
    }

    /// CA Route#grain_train_choices: route trains but the E-train and the
    /// pullman-attached one (train ids).
    pub(crate) fn g1822_ca_grain_choices(&self, c: &str) -> Vec<String> {
        let pullman = self.rnd().pullman_train.clone();
        self.g1822_route_trains(c)
            .into_iter()
            .filter(|t| t.name != self.fam().e_train && Some(&t.id) != pullman.as_ref())
            .map(|t| t.id)
            .collect()
    }

    /// CA Route#choosing?.
    pub(crate) fn g1822_ca_choosing(&self, c: &str) -> Option<CaChoosing> {
        let ci = self.ci(c);
        let r = &self.rnd().ca;
        let pullman = !r.skipped_pullman
            && self.rnd().pullman_train.is_none()
            && self.corporations[ci].trains.iter().any(|t| t.name == self.fam().extra_train_pullman)
            && !self.g1822_pullman_choices(c).is_empty();
        if pullman {
            return Some(CaChoosing::Pullman);
        }
        let grain = !r.skipped_grain
            && r.grain_train.is_none()
            && self.corporations[ci].trains.iter().any(Self::g1822_ca_grain_train)
            && !self.g1822_ca_grain_choices(c).is_empty();
        grain.then_some(CaChoosing::Grain)
    }

    /// CA Route#process_choose.
    pub(crate) fn g1822_ca_route_choose(&mut self, c: &str, choice: &str) -> Result<(), GameError> {
        let Some(choosing) = self.g1822_ca_choosing(c) else {
            return err("Unexpected value for @choosing when processing action \"choose\"");
        };
        if choice == CHOICE_SKIP {
            match choosing {
                CaChoosing::Pullman => self.rndm().ca.skipped_pullman = true,
                CaChoosing::Grain => self.rndm().ca.skipped_grain = true,
            }
            return Ok(());
        }
        let choices = match choosing {
            CaChoosing::Pullman => self.g1822_pullman_choices(c),
            CaChoosing::Grain => self.g1822_ca_grain_choices(c),
        };
        // choices[action.choice.to_i] (Ruby String#to_i; nil raises)
        let n = ruby_to_i(choice);
        let idx = if n < 0 { choices.len() as i64 + n } else { n };
        let tid = usize::try_from(idx)
            .ok()
            .and_then(|i| choices.get(i).cloned())
            .ok_or_else(|| GameError::new("bad train choice"))?;
        let ci = self.ci(c);
        let t = self.corporations[ci].trains.iter_mut().find(|t| t.id == tid).unwrap();
        match choosing {
            CaChoosing::Pullman => {
                // CA attach_pullman: "<name>+<2 × distance>"
                let orig = (t.name.clone(), t.distance);
                t.name = format!("{}+{}", t.name, 2 * t.distance);
                self.gm().round.pullman_original = Some(orig);
                self.gm().round.pullman_train = Some(tid);
            }
            CaChoosing::Grain => {
                // attach_grain_train: "<name>G" ("<name>-G" after a P)
                let orig = t.name.clone();
                t.name = if orig.ends_with('P') { format!("{orig}-G") } else { format!("{orig}G") };
                // (the memoized crowded_corps still counts it as `orig`)
                if !self.g().ca.crowd_memo_names.iter().any(|(id, _)| *id == tid) {
                    self.gm().ca.crowd_memo_names.push((tid.clone(), orig.clone()));
                }
                self.gm().round.ca.grain_original = Some(orig);
                self.gm().round.ca.grain_train = Some(tid);
            }
        }
        Ok(())
    }

    /// CA Route#process_run_routes: detach_grain_train.
    pub(crate) fn g1822_ca_detach_grain(&mut self) {
        let (Some(tid), Some(name)) = (self.rnd().ca.grain_train.clone(), self.rnd().ca.grain_original.clone()) else {
            return;
        };
        if let Some(t) = self.corporations.iter_mut().flat_map(|co| co.trains.iter_mut()).find(|t| t.id == tid) {
            t.name = name;
        }
        self.gm().ca.crowd_memo_names.retain(|(id, _)| *id != tid);
        self.rndm().ca.grain_train = None;
        self.rndm().ca.grain_original = None;
    }

    /// Trains moved: Ruby resets the memoized `crowded_corps` (Game#buy_train,
    /// #remove_train of a corporation's train, #transfer(:trains),
    /// #rust_trains!), which from now on counts every train under its name.
    pub(crate) fn g1822_trains_moved(&mut self) {
        if !self.g().ca.crowd_memo_names.is_empty() {
            self.gm().ca.crowd_memo_names.clear();
        }
    }

    /// The name the memoized `crowded_corps` counts a train under: its name
    /// before a grain attachment newer than the memo, else its name.
    pub(crate) fn g1822_crowd_memo_name<'a>(&'a self, t: &'a Train) -> &'a str {
        self.g()
            .ca
            .crowd_memo_names
            .iter()
            .find(|(id, _)| *id == t.id)
            .map_or(t.name.as_str(), |(_, n)| n.as_str())
    }

    /// CA Route#setup (each operator). Ruby drops `@grain_train` here
    /// without `detach_grain_train`: when attaching the grain train took a
    /// corporation's last local train (L / LP -> "LG" / "LP-G";
    /// `can_run_route?` asks for a LOCAL_TRAINS name) and it has no route on
    /// the graph, its Route step has nothing left to offer and is skipped —
    /// no `run_routes`, so no detach — and the train keeps the grain name
    /// (with `no_local`) for the rest of the game. Mirrored: the name stays;
    /// the router runs such a train as Ruby does (an L / LP that is not
    /// local, `g1822_train_kind_of`), the RL descriptors name its base type
    /// (`rl::g1822::train_type`), and it rusts as its base type.
    pub(crate) fn g1822_ca_route_setup(&mut self) {
        let r = &mut self.rndm().ca;
        r.skipped_pullman = false;
        r.skipped_grain = false;
        r.grain_train = None;
        r.grain_original = None;
    }

    /// `small_mail_subsidy` by phase.
    fn g1822_ca_small_mail_subsidy(&self) -> i32 {
        match self.phase_num() {
            3..=4 => 20,
            5..=6 => 30,
            7 => 40,
            _ => 0,
        }
    }

    /// CA `mail_contract_bonus` → its subsidy total, from G1822's large
    /// contract bonuses (best first, one per contract).
    pub(crate) fn g1822_ca_mail_subsidy(&self, c: &str, large: Vec<i32>) -> i32 {
        let owner = crate::entities::EntityId::corporation(c);
        let small_contracts =
            PRIVATE_SMALL_MAIL_CONTRACTS.iter().filter(|p| !self.co_closed(p) && self.co_owner(p) == owner).count();
        let all = large.len() + small_contracts;
        if all == 0 {
            return 0;
        }
        let routes = self.rnd().routes.len();
        let small: Vec<i32> = if small_contracts > 0 {
            vec![self.g1822_ca_small_mail_subsidy(); routes.min(small_contracts)]
        } else {
            Vec::new()
        };
        if routes >= all || large.is_empty() || small.is_empty() {
            return large.iter().sum::<i32>() + small.iter().sum::<i32>();
        }
        let (mut li, mut si) = (0usize, 0usize);
        let mut total = 0;
        for _ in 0..routes {
            let l = large.get(li).copied();
            let s = small.get(si).copied();
            if s.unwrap_or(0) > l.unwrap_or(0) {
                si += 1;
                total += s.unwrap_or(0);
            } else {
                li += 1;
                total += l.unwrap_or(0);
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use crate::game::BaseGame;

    /// `attach_grain_train` appends "-G" after a P and "G" otherwise; a name
    /// left on and attached again gets a second suffix.
    #[test]
    fn grain_suffixes_come_off() {
        for (name, base) in [
            ("LP-G", "LP"),
            ("2P-G", "2P"),
            ("5P-G", "5P"),
            ("LG", "L"),
            ("3G", "3"),
            ("LP-GG", "LP"),
            ("LGG", "L"),
            ("G", "G"),
            ("LP", "LP"),
            ("P", "P"),
            ("3+6", "3+6"),
            ("E", "E"),
        ] {
            assert_eq!(BaseGame::g1822_ca_ungrained(name), base, "{name}");
        }
    }
}
