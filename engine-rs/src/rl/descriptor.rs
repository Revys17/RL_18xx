//! The action-descriptor schema: what every policy index of every title
//! means, in title-independent terms (`docs/rl_descriptors.md`).
//!
//! A descriptor is one row of small integers. The global vocabularies here
//! (families, entity kinds, choices, price kinds, flags) are shared by every
//! title and are APPEND-ONLY: a trained model embeds their ids, so an id is
//! never reused or renumbered. Entity, hex and tile fields index the title's
//! own vocabularies (`super::layout::Vocab`), whose orders come from the
//! title data.

/// "No value" in every descriptor field.
pub const NONE: i32 = -1;

/// The descriptor columns, in the order of [`Descriptor::to_row`] (and of
/// the Python array's columns).
pub const FIELDS: [&str; 14] = [
    "family",
    "ent0_kind",
    "ent0",
    "ent1_kind",
    "ent1",
    "via",
    "hex",
    "city",
    "tile",
    "rot",
    "param",
    "choice",
    "price_kind",
    "flags",
];

/// Action families: one per kind of decision, shared by every title.
/// `id` is the stable vocabulary id (append-only).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    Pass,
    BidCompany,
    BidCorporation,
    Par,
    BuyShares,
    SellShares,
    RedeemShares,
    IssueShares,
    ExchangeShares,
    PlaceToken,
    RemoveToken,
    LayTile,
    BuyTrainDepot,
    BuyTrainPool,
    BuyTrainExchange,
    BuyTrainCorp,
    DiscardTrain,
    Dividend,
    BuyCompany,
    RunRoutes,
    Bankrupt,
    Merge,
    Convert,
    Choose,
    AcquireCompany,
    ChooseAbility,
    HexToken,
    Assign,
    PayoffPlayerDebt,
    PayoffPlayerDebtPartial,
}

/// Every family in id order (the id is the position).
pub const FAMILIES: [Family; 30] = [
    Family::Pass,
    Family::BidCompany,
    Family::BidCorporation,
    Family::Par,
    Family::BuyShares,
    Family::SellShares,
    Family::RedeemShares,
    Family::IssueShares,
    Family::ExchangeShares,
    Family::PlaceToken,
    Family::RemoveToken,
    Family::LayTile,
    Family::BuyTrainDepot,
    Family::BuyTrainPool,
    Family::BuyTrainExchange,
    Family::BuyTrainCorp,
    Family::DiscardTrain,
    Family::Dividend,
    Family::BuyCompany,
    Family::RunRoutes,
    Family::Bankrupt,
    Family::Merge,
    Family::Convert,
    Family::Choose,
    Family::AcquireCompany,
    Family::ChooseAbility,
    Family::HexToken,
    Family::Assign,
    Family::PayoffPlayerDebt,
    Family::PayoffPlayerDebtPartial,
];

impl Family {
    pub fn id(self) -> i32 {
        FAMILIES
            .iter()
            .position(|f| *f == self)
            .expect("every family is listed") as i32
    }

    pub fn name(self) -> &'static str {
        match self {
            Family::Pass => "Pass",
            Family::BidCompany => "Bid.company",
            Family::BidCorporation => "Bid.corporation",
            Family::Par => "Par",
            Family::BuyShares => "BuyShares",
            Family::SellShares => "SellShares",
            Family::RedeemShares => "RedeemShares",
            Family::IssueShares => "IssueShares",
            Family::ExchangeShares => "ExchangeShares",
            Family::PlaceToken => "PlaceToken",
            Family::RemoveToken => "RemoveToken",
            Family::LayTile => "LayTile",
            Family::BuyTrainDepot => "BuyTrain.depot",
            Family::BuyTrainPool => "BuyTrain.pool",
            Family::BuyTrainExchange => "BuyTrain.exchange",
            Family::BuyTrainCorp => "BuyTrain.corp",
            Family::DiscardTrain => "DiscardTrain",
            Family::Dividend => "Dividend",
            Family::BuyCompany => "BuyCompany",
            Family::RunRoutes => "RunRoutes",
            Family::Bankrupt => "Bankrupt",
            Family::Merge => "Merge",
            Family::Convert => "Convert",
            Family::Choose => "Choose",
            Family::AcquireCompany => "AcquireCompany",
            Family::ChooseAbility => "ChooseAbility",
            Family::HexToken => "HexToken",
            Family::Assign => "Assign",
            Family::PayoffPlayerDebt => "PayoffPlayerDebt",
            Family::PayoffPlayerDebtPartial => "PayoffPlayerDebtPartial",
        }
    }

    /// Map families: the policy head factors these over the map
    /// (P(hex) · P(tile | hex) · P(rot | hex, tile) for lays, P(hex) ·
    /// P(city | hex) for tokens) instead of scoring them flat.
    pub fn is_map_family(self) -> bool {
        matches!(
            self,
            Family::LayTile | Family::PlaceToken | Family::RemoveToken | Family::HexToken
        )
    }
}

/// What an entity field (`ent0` / `ent1`) points at: its kind selects the
/// title vocabulary the index is into (`Vocab::players` is the game's
/// seating order, the others are title data).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EntKind {
    Player,
    Corporation,
    Company,
    Train,
}

pub const ENT_KINDS: [EntKind; 4] = [EntKind::Player, EntKind::Corporation, EntKind::Company, EntKind::Train];

impl EntKind {
    pub fn id(self) -> i32 {
        ENT_KINDS.iter().position(|k| *k == self).expect("listed") as i32
    }

    pub fn name(self) -> &'static str {
        match self {
            EntKind::Player => "player",
            EntKind::Corporation => "corporation",
            EntKind::Company => "company",
            EntKind::Train => "train",
        }
    }
}

/// How an index's price is expressed. Never a fixed menu of prices: a
/// price-bearing index is ONE index; the legal range `[lo, hi]` (and its
/// step) comes from the state when the index is legal, and a price head
/// conditioned on the descriptor gives a distribution over that range
/// through `price_pmf`'s cells of the kind's `cell_type`. Indices without a
/// price kind cost what the state says (par cost, a depot train's face
/// value, a token's price) or nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PriceKind {
    /// An auction bid: `price_pmf` "Bid" cells (a ladder of
    /// `lo + bid_price_step * j` atoms plus one bin).
    Bid,
    /// A train bought from another corporation: "BuyTrain" cells (atoms at
    /// lo, hi and hi - 1, plus range bins).
    TrainTransfer,
    /// A private company bought by a corporation: "BuyCompany" cells
    /// (atoms at lo and hi, plus range bins).
    CompanyPurchase,
    /// An amount paid on a debt (1822 partial player-debt payoff): cells to
    /// be fixed from a human price census when the family is first used.
    Amount,
}

pub const PRICE_KINDS: [PriceKind; 4] = [
    PriceKind::Bid,
    PriceKind::TrainTransfer,
    PriceKind::CompanyPurchase,
    PriceKind::Amount,
];

impl PriceKind {
    pub fn id(self) -> i32 {
        PRICE_KINDS.iter().position(|k| *k == self).expect("listed") as i32
    }

    pub fn name(self) -> &'static str {
        match self {
            PriceKind::Bid => "bid",
            PriceKind::TrainTransfer => "train_transfer",
            PriceKind::CompanyPurchase => "company_purchase",
            PriceKind::Amount => "amount",
        }
    }

    /// The `price_pmf` cell layout this kind prices with (`None`: not
    /// fixed yet).
    pub fn cell_type(self) -> Option<&'static str> {
        match self {
            PriceKind::Bid => Some("Bid"),
            PriceKind::TrainTransfer => Some("BuyTrain"),
            PriceKind::CompanyPurchase => Some("BuyCompany"),
            PriceKind::Amount => None,
        }
    }
}

/// Categorical options of a family, one global vocabulary (append-only):
/// share sources, dividend kinds, named choices. A family uses at most one.
pub const CHOICES: [&str; 6] = [
    "source:ipo",
    "source:market",
    "dividend:payout",
    "dividend:withhold",
    "dividend:half",
    "choose:nationalize",
];

/// The choice id of a vocabulary string (panics on an unknown one: the
/// vocabulary is code, a missing entry is a bug).
pub fn choice_id(name: &str) -> i32 {
    CHOICES
        .iter()
        .position(|c| *c == name)
        .unwrap_or_else(|| panic!("no choice {name:?} in the descriptor vocabulary")) as i32
}

/// Flag bits.
pub mod flags {
    /// The index is never legal: a slot of a frozen layout that its index
    /// function never produces (1830's price slots of cross-corporation
    /// trains past the first, BuyCompany's "max" slots).
    pub const DEAD: i32 = 1;
    /// The train (type) of this index is decided by the state, not the
    /// index: 1830's single depot slot (the depot's next train) and its D
    /// trade-in (which owned train is traded).
    pub const TRAIN_FROM_STATE: i32 = 2;

    pub const ALL: [(&str, i32); 2] = [("dead", DEAD), ("train_from_state", TRAIN_FROM_STATE)];
}

/// One index's meaning. Every field is [`NONE`] unless the family uses it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Descriptor {
    /// [`Family::id`].
    pub family: i32,
    /// [`EntKind::id`] of `ent0` / `ent1`, and the entity's index in the
    /// title vocabulary of that kind.
    pub ent0_kind: i32,
    pub ent0: i32,
    pub ent1_kind: i32,
    pub ent1: i32,
    /// The company whose ability performs the action (company vocabulary
    /// index); NONE when the acting player or corporation acts itself.
    pub via: i32,
    /// Hex vocabulary index (map order), the city on that hex's tile, tile
    /// vocabulary index, rotation 0-5.
    pub hex: i32,
    pub city: i32,
    pub tile: i32,
    pub rot: i32,
    /// The family's number (Par: the par price in $; Buy/Sell/Redeem
    /// shares: the percent moved).
    pub param: i32,
    /// An entry of [`CHOICES`].
    pub choice: i32,
    /// [`PriceKind::id`], NONE for indices without a price parameter.
    pub price_kind: i32,
    /// [`flags`] bits.
    pub flags: i32,
}

impl Descriptor {
    pub fn new(family: Family) -> Self {
        Descriptor {
            family: family.id(),
            ent0_kind: NONE,
            ent0: NONE,
            ent1_kind: NONE,
            ent1: NONE,
            via: NONE,
            hex: NONE,
            city: NONE,
            tile: NONE,
            rot: NONE,
            param: NONE,
            choice: NONE,
            price_kind: NONE,
            flags: 0,
        }
    }

    pub fn to_row(&self) -> [i32; 14] {
        [
            self.family,
            self.ent0_kind,
            self.ent0,
            self.ent1_kind,
            self.ent1,
            self.via,
            self.hex,
            self.city,
            self.tile,
            self.rot,
            self.param,
            self.choice,
            self.price_kind,
            self.flags,
        ]
    }

    pub fn family_enum(&self) -> Family {
        FAMILIES[self.family as usize]
    }

    pub fn is_dead(&self) -> bool {
        self.flags & flags::DEAD != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The global vocabularies are append-only: a trained model embeds
    /// these ids. Pin the ones every layout so far uses.
    #[test]
    fn global_vocab_ids_are_pinned() {
        let fam: Vec<&str> = FAMILIES.iter().map(|f| f.name()).collect();
        assert_eq!(
            fam,
            vec![
                "Pass",
                "Bid.company",
                "Bid.corporation",
                "Par",
                "BuyShares",
                "SellShares",
                "RedeemShares",
                "IssueShares",
                "ExchangeShares",
                "PlaceToken",
                "RemoveToken",
                "LayTile",
                "BuyTrain.depot",
                "BuyTrain.pool",
                "BuyTrain.exchange",
                "BuyTrain.corp",
                "DiscardTrain",
                "Dividend",
                "BuyCompany",
                "RunRoutes",
                "Bankrupt",
                "Merge",
                "Convert",
                "Choose",
                "AcquireCompany",
                "ChooseAbility",
                "HexToken",
                "Assign",
                "PayoffPlayerDebt",
                "PayoffPlayerDebtPartial",
            ]
        );
        for (i, f) in FAMILIES.iter().enumerate() {
            assert_eq!(f.id(), i as i32);
        }
        let kinds: Vec<&str> = ENT_KINDS.iter().map(|k| k.name()).collect();
        assert_eq!(kinds, vec!["player", "corporation", "company", "train"]);
        let prices: Vec<&str> = PRICE_KINDS.iter().map(|k| k.name()).collect();
        assert_eq!(prices, vec!["bid", "train_transfer", "company_purchase", "amount"]);
        assert_eq!(
            CHOICES[..4],
            ["source:ipo", "source:market", "dividend:payout", "dividend:withhold"]
        );
        assert_eq!(choice_id("dividend:half"), 4);
        assert_eq!(choice_id("choose:nationalize"), 5);
        assert_eq!(flags::DEAD, 1);
        assert_eq!(flags::TRAIN_FROM_STATE, 2);
        // Every price kind used by a layout prices through price_pmf cells.
        for k in [PriceKind::Bid, PriceKind::TrainTransfer, PriceKind::CompanyPurchase] {
            assert!(crate::price_pmf::layout(k.cell_type().unwrap()).is_some(), "{k:?}");
        }
    }
}
