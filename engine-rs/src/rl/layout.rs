//! The block registry: a title's action layout as an ordered list of
//! blocks, each one action family over a tuple of axes whose entries come
//! from the title data (`docs/rl_descriptors.md`, design §2.2).
//!
//! `index = block.offset + mixed_radix(axis positions)`, the first axis
//! varying slowest. Every index gets a [`Descriptor`] built from its
//! block's family and its position on each axis; an axis with one entry
//! pins a field for the whole block (e.g. the company of a company-ability
//! block). The layout's fingerprint hashes the blocks, the title
//! vocabularies and every descriptor, so any change to what an index means
//! shows up as a fingerprint change (pinned per title by a test).

use super::descriptor::{choice_id, flags, Descriptor, EntKind, Family, PriceKind, FIELDS, NONE};
use crate::title::{CorpType, GameTitle};

/// A title's vocabularies: the descriptor's entity, hex and tile fields are
/// positions in these lists. All come from the title data, in its order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vocab {
    /// `GameTitle::corporations()` order (every kind: majors, minors,
    /// nationals).
    pub corporations: Vec<&'static str>,
    /// `GameTitle::companies()` order (auctionable or not).
    pub companies: Vec<&'static str>,
    /// `GameTitle::trains()` names in order, variants included (1822's "2"
    /// is a variant of the L card).
    pub trains: Vec<&'static str>,
    /// `GameTitle::hex_definitions()` order = `BaseGame.hexes` order.
    pub hexes: Vec<&'static str>,
    /// `GameTitle::tile_counts()` order (the title's tile manifest).
    pub tiles: Vec<&'static str>,
}

impl Vocab {
    pub fn for_title(title: &dyn GameTitle) -> Self {
        let mut trains: Vec<&'static str> = Vec::new();
        for t in title.trains() {
            if !trains.contains(&t.name) {
                trains.push(t.name);
            }
        }
        Vocab {
            corporations: title.corporations().iter().map(|c| c.sym).collect(),
            companies: title.companies().iter().map(|c| c.sym).collect(),
            trains,
            hexes: title.hex_definitions().iter().map(|h| h.coord).collect(),
            tiles: title.tile_counts().iter().map(|(t, _)| *t).collect(),
        }
    }

    fn list(&self, kind: EntKind) -> Option<&[&'static str]> {
        match kind {
            EntKind::Corporation => Some(&self.corporations),
            EntKind::Company => Some(&self.companies),
            EntKind::Train => Some(&self.trains),
            // Player axes index the game's seating order, not title data.
            EntKind::Player => None,
        }
    }

    fn position(list: &[&str], sym: &str) -> Option<i32> {
        list.iter().position(|x| *x == sym).map(|p| p as i32)
    }

    pub fn entity(&self, kind: EntKind, sym: &str) -> Option<i32> {
        Self::position(self.list(kind)?, sym)
    }
    pub fn hex(&self, coord: &str) -> Option<i32> {
        Self::position(&self.hexes, coord)
    }
    pub fn tile(&self, name: &str) -> Option<i32> {
        Self::position(&self.tiles, name)
    }
}

/// One axis of a block: which descriptor field it fills and its entries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Axis {
    /// An entity field: `slot` 0 fills `ent0`, 1 fills `ent1`.
    Ent {
        slot: u8,
        kind: EntKind,
        syms: Vec<&'static str>,
    },
    /// Seats (`ent{slot}` of kind player): seat positions 0..n.
    Seats {
        slot: u8,
        n: usize,
    },
    /// The company acting through its ability (`via`).
    Via(Vec<&'static str>),
    Hex(Vec<&'static str>),
    /// City slots `(hex, city on the hex's tile)`: fills `hex` and `city`.
    City(Vec<(&'static str, usize)>),
    Tile(Vec<&'static str>),
    /// Rotations 0-5.
    Rotation,
    /// Numbers (`param`).
    Param(Vec<i32>),
    /// Entries of the global choice vocabulary (`choice`).
    Choice(Vec<&'static str>),
    /// A frozen layout's legacy price slots: only the first entry is ever
    /// legal (the price itself comes from the price head), the others are
    /// [`flags::DEAD`].
    LegacyPrice(Vec<&'static str>),
}

impl Axis {
    pub fn len(&self) -> usize {
        match self {
            Axis::Ent { syms, .. } => syms.len(),
            Axis::Seats { n, .. } => *n,
            Axis::Via(v) | Axis::Hex(v) | Axis::Tile(v) | Axis::Choice(v) | Axis::LegacyPrice(v) => v.len(),
            Axis::City(v) => v.len(),
            Axis::Rotation => 6,
            Axis::Param(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The axis's name (`ent0:company`, `hex`, ...).
    pub fn name(&self) -> String {
        match self {
            Axis::Ent { slot, kind, .. } => format!("ent{slot}:{}", kind.name()),
            Axis::Seats { slot, .. } => format!("ent{slot}:player"),
            Axis::Via(_) => "via".into(),
            Axis::Hex(_) => "hex".into(),
            Axis::City(_) => "city".into(),
            Axis::Tile(_) => "tile".into(),
            Axis::Rotation => "rot".into(),
            Axis::Param(_) => "param".into(),
            Axis::Choice(_) => "choice".into(),
            Axis::LegacyPrice(_) => "legacy_price".into(),
        }
    }

    /// The entries as strings (fingerprint, Python introspection).
    pub fn entries(&self) -> Vec<String> {
        match self {
            Axis::Ent { syms, .. } => syms.iter().map(|s| s.to_string()).collect(),
            Axis::Seats { n, .. } => (0..*n).map(|i| i.to_string()).collect(),
            Axis::Via(v) | Axis::Hex(v) | Axis::Tile(v) | Axis::Choice(v) | Axis::LegacyPrice(v) => {
                v.iter().map(|s| s.to_string()).collect()
            }
            Axis::City(v) => v.iter().map(|(h, c)| format!("{h}.{c}")).collect(),
            Axis::Rotation => (0..6).map(|r| r.to_string()).collect(),
            Axis::Param(v) => v.iter().map(|p| p.to_string()).collect(),
        }
    }

    // -- axes from title data (the sources new titles' layouts use) --

    /// Companies in title order; `auctionable_only` keeps the ones the
    /// opening auction offers (1867 hides its "3" phase blocker).
    pub fn companies(slot: u8, title: &dyn GameTitle, auctionable_only: bool) -> Axis {
        let syms = title
            .companies()
            .iter()
            .filter(|c| c.auctionable || !auctionable_only)
            .map(|c| c.sym)
            .collect();
        Axis::Ent {
            slot,
            kind: EntKind::Company,
            syms,
        }
    }

    /// Corporations in title order, of one kind or all.
    pub fn corporations(slot: u8, title: &dyn GameTitle, kind: Option<CorpType>) -> Axis {
        let syms = title
            .corporations()
            .iter()
            .filter(|c| kind.map_or(true, |k| c.corp_type == k))
            .map(|c| c.sym)
            .collect();
        Axis::Ent {
            slot,
            kind: EntKind::Corporation,
            syms,
        }
    }

    /// Train names in title order (variants included).
    pub fn trains(slot: u8, title: &dyn GameTitle) -> Axis {
        Axis::Ent {
            slot,
            kind: EntKind::Train,
            syms: Vocab::for_title(title).trains,
        }
    }

    /// Every map hex, in map order.
    pub fn map_hexes(title: &dyn GameTitle) -> Axis {
        Axis::Hex(title.hex_definitions().iter().map(|h| h.coord).collect())
    }

    /// The title's tile manifest.
    pub fn tiles(title: &dyn GameTitle) -> Axis {
        Axis::Tile(title.tile_counts().iter().map(|(t, _)| *t).collect())
    }

    /// The market's par prices for a corporation kind (the cells of the
    /// title's par types for that kind), ascending, deduplicated.
    pub fn par_prices(title: &dyn GameTitle, kind: CorpType) -> Axis {
        let market = crate::core::StockMarket::new(title.market_grid(), title.market_movement());
        let mut prices: Vec<i32> = market
            .share_prices_with_types(title.par_price_types(kind))
            .iter()
            .map(|sp| sp.price)
            .collect();
        prices.sort_unstable();
        prices.dedup();
        Axis::Param(prices)
    }
}

/// A block before it is placed: its family, axes, price kind and flags.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockSpec {
    /// The block's name (unique in the layout).
    pub name: String,
    pub family: Family,
    pub axes: Vec<Axis>,
    pub price: Option<PriceKind>,
    /// Flag bits set on every index of the block.
    pub flags: i32,
}

impl BlockSpec {
    pub fn new(name: impl Into<String>, family: Family, axes: Vec<Axis>) -> Self {
        BlockSpec {
            name: name.into(),
            family,
            axes,
            price: None,
            flags: 0,
        }
    }

    pub fn priced(mut self, kind: PriceKind) -> Self {
        self.price = Some(kind);
        self
    }

    pub fn flagged(mut self, bits: i32) -> Self {
        self.flags |= bits;
        self
    }

    pub fn size(&self) -> u32 {
        self.axes.iter().map(|a| a.len() as u32).product()
    }
}

/// A placed block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub spec: BlockSpec,
    pub offset: u32,
    pub size: u32,
}

/// A title's action layout: blocks, descriptors, vocabularies, fingerprint.
#[derive(Clone, Debug)]
pub struct RlLayout {
    pub title: &'static str,
    /// The layout version (`"1830-v1"` for the frozen 1830 layout). A
    /// change that moves any index's meaning needs a new version.
    pub version: String,
    pub vocab: Vocab,
    pub blocks: Vec<Block>,
    pub descriptors: Vec<Descriptor>,
    pub total: u32,
    pub fingerprint: u64,
}

impl RlLayout {
    /// Place the blocks contiguously, in order, and derive every index's
    /// descriptor. Errs on an axis entry the title vocabulary lacks, an
    /// empty block or a repeated block name.
    pub fn build(
        title: &'static str,
        version: impl Into<String>,
        vocab: Vocab,
        specs: Vec<BlockSpec>,
    ) -> Result<RlLayout, String> {
        let mut blocks: Vec<Block> = Vec::new();
        let mut descriptors: Vec<Descriptor> = Vec::new();
        let mut offset: u32 = 0;
        for spec in specs {
            if blocks.iter().any(|b| b.spec.name == spec.name) {
                return Err(format!("{title}: block {} repeated", spec.name));
            }
            let size = spec.size();
            if size == 0 {
                return Err(format!("{title}: block {} is empty", spec.name));
            }
            for j in 0..size {
                descriptors
                    .push(Self::describe(&vocab, &spec, j).map_err(|e| format!("{title}: block {}: {e}", spec.name))?);
            }
            blocks.push(Block { spec, offset, size });
            offset += size;
        }
        let mut layout = RlLayout {
            title,
            version: version.into(),
            vocab,
            blocks,
            descriptors,
            total: offset,
            fingerprint: 0,
        };
        layout.fingerprint = layout.compute_fingerprint();
        Ok(layout)
    }

    /// The descriptor of position `j` of a block (mixed radix, first axis
    /// slowest).
    fn describe(vocab: &Vocab, spec: &BlockSpec, j: u32) -> Result<Descriptor, String> {
        let mut d = Descriptor::new(spec.family);
        d.price_kind = spec.price.map_or(NONE, |k| k.id());
        d.flags = spec.flags;
        let mut rest = j as usize;
        let mut positions = vec![0usize; spec.axes.len()];
        for (k, axis) in spec.axes.iter().enumerate().rev() {
            positions[k] = rest % axis.len();
            rest /= axis.len();
        }
        for (axis, &p) in spec.axes.iter().zip(&positions) {
            match axis {
                Axis::Ent { slot, kind, syms } => {
                    let i = vocab
                        .entity(*kind, syms[p])
                        .ok_or_else(|| format!("no {} {:?}", kind.name(), syms[p]))?;
                    Self::set_ent(&mut d, *slot, kind.id(), i);
                }
                Axis::Seats { slot, .. } => Self::set_ent(&mut d, *slot, EntKind::Player.id(), p as i32),
                Axis::Via(v) => {
                    d.via = vocab
                        .entity(EntKind::Company, v[p])
                        .ok_or_else(|| format!("no company {:?}", v[p]))?;
                }
                Axis::Hex(v) => d.hex = vocab.hex(v[p]).ok_or_else(|| format!("no hex {:?}", v[p]))?,
                Axis::City(v) => {
                    d.hex = vocab.hex(v[p].0).ok_or_else(|| format!("no hex {:?}", v[p].0))?;
                    d.city = v[p].1 as i32;
                }
                Axis::Tile(v) => d.tile = vocab.tile(v[p]).ok_or_else(|| format!("no tile {:?}", v[p]))?,
                Axis::Rotation => d.rot = p as i32,
                Axis::Param(v) => d.param = v[p],
                Axis::Choice(v) => d.choice = choice_id(v[p]),
                Axis::LegacyPrice(_) => {
                    if p > 0 {
                        d.flags |= flags::DEAD;
                    }
                }
            }
        }
        Ok(d)
    }

    fn set_ent(d: &mut Descriptor, slot: u8, kind: i32, i: i32) {
        if slot == 0 {
            d.ent0_kind = kind;
            d.ent0 = i;
        } else {
            d.ent1_kind = kind;
            d.ent1 = i;
        }
    }

    pub fn descriptor(&self, idx: u32) -> Option<&Descriptor> {
        self.descriptors.get(idx as usize)
    }

    /// The block holding `idx`.
    pub fn block_of(&self, idx: u32) -> Option<&Block> {
        if idx >= self.total {
            return None;
        }
        let i = self.blocks.partition_point(|b| b.offset <= idx);
        self.blocks.get(i.checked_sub(1)?)
    }

    pub fn block(&self, name: &str) -> Option<&Block> {
        self.blocks.iter().find(|b| b.spec.name == name)
    }

    /// The canonical text the fingerprint hashes: version, descriptor
    /// fields, the title vocabularies, every block (name, family, offset,
    /// size, price kind, flags, axes with their entries) and every
    /// descriptor row. Global vocabulary ids enter through the rows, their
    /// names through the blocks; appending to a global vocabulary changes
    /// neither for an existing layout.
    pub fn canonical_text(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "rl-layout\n{}\n{}\nfields:{}\n",
            self.title,
            self.version,
            FIELDS.join(",")
        ));
        s.push_str(&format!("corporations:{}\n", self.vocab.corporations.join(",")));
        s.push_str(&format!("companies:{}\n", self.vocab.companies.join(",")));
        s.push_str(&format!("trains:{}\n", self.vocab.trains.join(",")));
        s.push_str(&format!("hexes:{}\n", self.vocab.hexes.join(",")));
        s.push_str(&format!("tiles:{}\n", self.vocab.tiles.join(",")));
        for b in &self.blocks {
            s.push_str(&format!(
                "block:{}|{}|{}|{}|{}|{}",
                b.spec.name,
                b.spec.family.name(),
                b.offset,
                b.size,
                b.spec.price.map_or("-", |k| k.name()),
                b.spec.flags
            ));
            for a in &b.spec.axes {
                s.push_str(&format!("|{}[{}]", a.name(), a.entries().join(",")));
            }
            s.push('\n');
        }
        for d in &self.descriptors {
            let row: Vec<String> = d.to_row().iter().map(|v| v.to_string()).collect();
            s.push_str(&row.join(","));
            s.push('\n');
        }
        s
    }

    fn compute_fingerprint(&self) -> u64 {
        fnv1a64(self.canonical_text().as_bytes())
    }

    pub fn fingerprint_hex(&self) -> String {
        format!("{:016x}", self.fingerprint)
    }
}

/// FNV-1a, 64-bit: a fixed, dependency-free hash (std's hashers promise no
/// stability across releases).
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_vocab() -> Vocab {
        Vocab {
            corporations: vec!["A", "B"],
            companies: vec!["X", "Y", "Z"],
            trains: vec!["2", "3"],
            hexes: vec!["A1", "A3", "B2"],
            tiles: vec!["7", "8", "9"],
        }
    }

    #[test]
    fn blocks_are_contiguous_and_mixed_radix_runs_the_last_axis_fastest() {
        let specs = vec![
            BlockSpec::new("Pass", Family::Pass, vec![]),
            BlockSpec::new(
                "BuyTrain.corp",
                Family::BuyTrainCorp,
                vec![
                    Axis::Ent {
                        slot: 0,
                        kind: EntKind::Corporation,
                        syms: vec!["A", "B"],
                    },
                    Axis::Ent {
                        slot: 1,
                        kind: EntKind::Train,
                        syms: vec!["2", "3"],
                    },
                ],
            )
            .priced(PriceKind::TrainTransfer),
            BlockSpec::new(
                "LayTile",
                Family::LayTile,
                vec![Axis::Hex(vec!["A3", "B2"]), Axis::Tile(vec!["9"]), Axis::Rotation],
            ),
        ];
        let lo = RlLayout::build("TINY", "tiny-v1", tiny_vocab(), specs).unwrap();
        assert_eq!(lo.total, 1 + 4 + 12);
        assert_eq!(
            lo.blocks.iter().map(|b| (b.offset, b.size)).collect::<Vec<_>>(),
            vec![(0, 1), (1, 4), (5, 12)]
        );
        // Pass: nothing but the family.
        assert_eq!(lo.descriptors[0], Descriptor::new(Family::Pass));
        // BuyTrain.corp [B, 2] is position 2 of its block (corp slowest).
        let d = lo.descriptors[3];
        assert_eq!(
            (d.family, d.ent0_kind, d.ent0, d.ent1_kind, d.ent1),
            (Family::BuyTrainCorp.id(), 1, 1, 3, 0)
        );
        assert_eq!(d.price_kind, PriceKind::TrainTransfer.id());
        // LayTile [B2, 9, rot 4] = 5 + 1*6 + 0*6 + 4.
        let d = lo.descriptors[5 + 6 + 4];
        assert_eq!((d.hex, d.tile, d.rot, d.city), (2, 2, 4, NONE));
        assert_eq!(lo.block_of(0).unwrap().spec.name, "Pass");
        assert_eq!(lo.block_of(4).unwrap().spec.name, "BuyTrain.corp");
        assert_eq!(lo.block_of(16).unwrap().spec.name, "LayTile");
        assert!(lo.block_of(17).is_none());
    }

    #[test]
    fn unknown_entries_empty_and_repeated_blocks_are_refused() {
        let bad = vec![BlockSpec::new(
            "Bid",
            Family::BidCompany,
            vec![Axis::Ent {
                slot: 0,
                kind: EntKind::Company,
                syms: vec!["Q"],
            }],
        )];
        assert!(RlLayout::build("TINY", "v", tiny_vocab(), bad)
            .unwrap_err()
            .contains("no company"));
        let empty = vec![BlockSpec::new(
            "Bid",
            Family::BidCompany,
            vec![Axis::Ent {
                slot: 0,
                kind: EntKind::Company,
                syms: vec![],
            }],
        )];
        assert!(RlLayout::build("TINY", "v", tiny_vocab(), empty)
            .unwrap_err()
            .contains("empty"));
        let twice = vec![
            BlockSpec::new("Pass", Family::Pass, vec![]),
            BlockSpec::new("Pass", Family::Pass, vec![]),
        ];
        assert!(RlLayout::build("TINY", "v", tiny_vocab(), twice)
            .unwrap_err()
            .contains("repeated"));
    }

    #[test]
    fn the_fingerprint_sees_every_change_of_meaning() {
        let base = || {
            vec![BlockSpec::new(
                "Par",
                Family::Par,
                vec![
                    Axis::Ent {
                        slot: 0,
                        kind: EntKind::Corporation,
                        syms: vec!["A", "B"],
                    },
                    Axis::Param(vec![67, 71]),
                ],
            )]
        };
        let a = RlLayout::build("TINY", "v1", tiny_vocab(), base()).unwrap();
        let b = RlLayout::build("TINY", "v1", tiny_vocab(), base()).unwrap();
        assert_eq!(a.fingerprint, b.fingerprint, "deterministic");
        let mut prices = base();
        prices[0].axes[1] = Axis::Param(vec![67, 72]);
        assert_ne!(
            a.fingerprint,
            RlLayout::build("TINY", "v1", tiny_vocab(), prices).unwrap().fingerprint
        );
        let mut order = base();
        order[0].axes[0] = Axis::Ent {
            slot: 0,
            kind: EntKind::Corporation,
            syms: vec!["B", "A"],
        };
        assert_ne!(
            a.fingerprint,
            RlLayout::build("TINY", "v1", tiny_vocab(), order).unwrap().fingerprint
        );
        let mut vocab = tiny_vocab();
        vocab.corporations.reverse();
        assert_ne!(
            a.fingerprint,
            RlLayout::build("TINY", "v1", vocab, base()).unwrap().fingerprint
        );
        assert_ne!(
            a.fingerprint,
            RlLayout::build("TINY", "v2", tiny_vocab(), base()).unwrap().fingerprint
        );
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    }

    /// Axes from title data: 1830's lists equal the flat layout's (which
    /// derives the same lists), and 1867's give the design's §2.3 sizes.
    #[test]
    fn axes_resolve_from_title_data() {
        let t30 = crate::title::resolve("1830");
        let lo = crate::action_index::layout_for("1830");
        assert_eq!(
            Axis::companies(0, t30, true).entries(),
            lo.company_offsets.iter().map(|s| s.to_string()).collect::<Vec<_>>()
        );
        assert_eq!(
            Axis::corporations(0, t30, None).entries(),
            lo.corporation_offsets.iter().map(|s| s.to_string()).collect::<Vec<_>>()
        );
        assert_eq!(
            Axis::trains(0, t30).entries(),
            lo.train_type_offsets.iter().map(|s| s.to_string()).collect::<Vec<_>>()
        );
        assert_eq!(
            Axis::par_prices(t30, CorpType::Major),
            Axis::Param(lo.par_price_offsets.clone())
        );
        assert_eq!(Axis::map_hexes(t30).len(), 93);
        // The frozen tile order is a permutation of the manifest.
        let mut manifest = Axis::tiles(t30).entries();
        let mut frozen: Vec<String> = lo.tile_offsets.iter().map(|s| s.to_string()).collect();
        manifest.sort();
        frozen.sort();
        assert_eq!(manifest, frozen);

        let t67 = crate::title::resolve("1867");
        assert_eq!(
            Axis::companies(0, t67, true).entries(),
            vec!["C&SL", "NFB", "MB", "QB", "SCT"]
        );
        assert_eq!(
            Axis::companies(0, t67, false).len(),
            6,
            "the hidden '3' blocker is a company"
        );
        assert_eq!(Axis::corporations(0, t67, Some(CorpType::Minor)).len(), 16);
        assert_eq!(Axis::corporations(0, t67, Some(CorpType::Major)).len(), 8);
        assert_eq!(
            Axis::trains(0, t67).entries(),
            vec!["2", "3", "4", "5", "6", "7", "8", "2+2", "5+5E"]
        );
    }
}
