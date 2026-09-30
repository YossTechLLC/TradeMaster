# Replacing MacroHFT's synthetic order-book features with real data

## 1. The problem, measured on the BTCUSDT fixture (`box/bench/build_data.py`)

Binance's free spot archive has no order-book history, so `build_data.py` invents a 5-level book: random
lognormal sizes and a random half-spread around the real close. Of MacroHFT's 36 per-bar inputs:

**Pure noise (12)**: random draws with no relation to the market:

| # | Feature | Upstream meaning | In the fixture |
|---|---|---|---|
| 1-5 | `bid1_size_n` … `bid5_size_n` | share of the bid depth (levels 1-5) at each level | `lognormal / sum`: random |
| 6-10 | `ask1_size_n` … `ask5_size_n` | same for asks | random |
| 11 | `wap_balance` | \|WAP(level 1) − WAP(level 2)\| | built from random sizes |
| 12 | `price_spread` | relative bid-ask spread, 2(a1−b1)/(a1+b1) | random half-spread |

**Constant (2)**: zero information:

| # | Feature | Upstream meaning | In the fixture |
|---|---|---|---|
| 13 | `buy_spread` | price distance bid1 → bid5 (how far 5 levels reach) | always 0.04 |
| 14 | `sell_spread` | price distance ask1 → ask5 | always 0.04 |

The same contamination reaches further:
- **Context features:** `buy_spread_trend_60` and `sell_spread_trend_60` are constant too (2 of the 9 inputs).
- **Duplicates:** the 5 book-price returns (`log_return_{bid1,bid2,ask1,ask2,wap_1}_price`) correlate **+1.000** with the close return. They are 5 copies of 1 real signal.
- **Price levels:** `wap_1` and `wap_2` are the close plus noise. They are also raw, non-stationary price levels.

## 2. What exists (research, 2026-09-29; sources in section 6)

- **No free historical per-level book exists for Binance, and no spot book data at all.** Binance stopped its tick-level history service in Nov 2024 and says it has no plans to restore it. Spot depth was answered "not available" (binance-public-data #300, #437, PR #348).
- **Free, real, futures only (USDⓈ-M BTCUSDT perpetual):**
  - `bookDepth`: cumulative depth within ±1, 2, 3, 4, 5 % of mid (±0.2 % added 2026-01-16), about every 30 s. It covers 2023-01-01 to now, with 3 missing days and ~41 partial files.
  - `bookTicker`: best bid/ask and sizes, tick by tick, **2023-05-16 → 2024-03-30 only**.
  - `aggTrades`/`trades`: exact aggressor side (`is_buyer_maker`), from 2019-12-31.
  - Spot `aggTrades`: from 2017-08-17.
- **What the MacroHFT/EarnHFT authors used:**
  - EarnHFT's own download code pulls **Tardis.dev `book_snapshot_5`, Binance spot**. MacroHFT (same group, 5-level minute LOB, Binance 2022-2023) almost certainly used the same pipeline.
  - Tardis serves the **first day of every month free**; verified by download (2019-03-30 onward).
  - Full history is paid. Tardis Spot costs, per month: Academic $450, Solo $900, Pro $1,350. Yearly billing gives 4 years back; everything since 2019 needs Business, about $42k per year.
  - Crypto Lake has 20-level Binance spot/futures books from 2022-11-14: $80/mo for individuals (no companies), $500/mo for companies. Kaiko and Amberdata sell 1-min snapshots on quote.
- **Established estimators when the book is unavailable:**
  - **Spread from OHLC:** Corwin & Schultz (2012 JF), Abdi & Ranaldo (2017 RFS), and **EDGE** (Ardia, Guidotti & Kroencke 2024 JFE). EDGE has the lowest bias and variance and the highest correlation with TAQ effective spreads, and has a maintained package: `pip install bidask`, `edge_rolling`.
  - **Validation on crypto:** Brauneis et al. (2021 JBF) checked CS/AR against real BTC/ETH books. They track the time variation of spreads and price impact (ρ 0.54-0.75 and 0.86), but not their level.
  - **Depth and impact:** Kyle's λ (returns on signed flow; per Cont, Kukanov & Stoikov 2014, the impact slope is inversely proportional to depth) and Amihud (2002). They are the AFML ch. 19 "microstructural features".
  - **The same feature set on Binance 1-minute bars:** Easley, O'Hara, Yang & Zhang (2024).
  - **Trade-flow imbalance** explains crypto price changes better than quote-based OFI (Silantyev 2019, BitMEX).
- **What has no trade-only equivalent in the literature:** the per-level size shares and level-1/level-2 WAPs. They need a book.

## 3. Method per feature, in three tiers (best first)

**Tier A: the real top-5 book, as the paper authors had it.**
Buy Tardis `book_snapshot_5` for Binance spot BTCUSDT (or Crypto Lake). Resample to bars: the last snapshot at or before each bar close, plus the bar mean for bars of 5 min and up. Compute all 14 features with the **upstream definitions unchanged** (`EarnHFT/data_preprocess/preprocess/create_feature.py`, `create_features_order_book`). Nothing is approximated, and the log returns and WAPs become real again.

**Tier B: free and real, but the instrument becomes the USDⓈ-M perpetual** (klines, aggTrades and `bookDepth`, all from the same instrument, 2023-01 → now). The book features are redefined on percent bands, because Binance publishes no levels:

| Upstream feature | Tier-B replacement (real data) | Basis |
|---|---|---|
| `bid{k}_size_n`, k = 1..5 | share of bid depth in band k: (D_bid(k%) − D_bid((k−1)%)) / D_bid(5%) | same "share of the side's depth per level", with price-distance bands for levels (depth-profile shape, cf. Cao, Hansch & Wang 2009) |
| `ask{k}_size_n`, k = 1..5 | same on the ask side | as above |
| `buy_spread` | 1 % ÷ D_bid(1 %): price distance per BTC of bid depth | inverse near-book depth, the quantity `buy_spread` measures |
| `sell_spread` | 1 % ÷ D_ask(1 %) | as above |
| `wap_balance` | \|I(1 %) − I(5 %)\|, where I(x) = (D_bid − D_ask)/(D_bid + D_ask): near-book vs deep-book imbalance | volume-imbalance signals (Cartea, Donnelly & Jaimungal 2018); the "near minus deeper level" contrast of `wap_balance` |
| `price_spread` | EDGE relative spread, `bidask.edge_rolling` over 60 bars of real OHLC | Ardia et al. 2024; validated against the real `bookTicker` spread for 2023-05 → 2024-03 before use |

- **Aggregation:** for each bar, take the last `bookDepth` snapshot at or before the close (1m), or the mean of the bar's snapshots (≥ 5m).
- **Gaps:** forward-fill up to 2 minutes; beyond that, drop the bar from training.
- **Band set:** use only ±1..5 % for a consistent feature set over the whole period (±0.2 % starts only in 2026).
- **Caveat:** ±1 % is about $800 on BTC, so these describe the *deep* book, not the first five queue levels. They are real liquidity shape, but not the same object as upstream.

**Tier C: spot, free, with no book.** Leave the per-level features out instead of padding (MacroHFT sizes its networks from the feature-list length, so 36 → fewer inputs works). Add the established trade/bar microstructure set on real spot aggTrades and klines:

| Replaces | Feature | Basis |
|---|---|---|
| `price_spread` | EDGE rolling relative spread | Ardia et al. 2024 (+ CS/AR, Brauneis et al. 2021) |
| `buy_spread` / `sell_spread` | Kyle's λ: rolling regression of returns on exact signed taker volume; Amihud illiquidity | Kyle 1985, Amihud 2002, AFML ch. 19; Cont et al. 2014 (λ ∝ 1/depth) |
| `wap_balance` | Roll measure / Roll impact | AFML ch. 19; the most important features in Easley et al. 2024 on Binance 1m |
| the 10 `size_n` | **dropped**: no trade-only equivalent exists. Optionally add VPIN (exact aggressor side). Andersen & Bondarenko (2014) found it has no incremental forecasting power beyond trading intensity, so it must earn its place (section 4). | literature |

**In every tier:**
- Collapse the 5 duplicate log returns into the one real close return. In Tier A, keep the bid/ask returns, which are real there.
- Replace `wap_1`/`wap_2` (price levels) with the close in Tier B/C; `buy_vwap`/`sell_vwap` already carry the real trade prices.
- Recompute `*_spread_trend_60` from the replacement features.

## 4. Validation before training: prove each replacement carries real signal

1. **Ground truth for spreads (Tier B/C):** on 2023-05-16 → 2024-03-30, compare the EDGE/CS/AR estimates with the real `bookTicker` quoted spread, and with the effective spread from aggTrades against the prevailing quote. Report the time-series correlation per bar size, as Brauneis et al. do. Keep an estimator only if ρ is at least about 0.5. BTC's quoted spread is nearly always one tick, so the effective spread is the meaningful target.
2. **Ground truth for the book (all tiers):** download the ~80 free Tardis first-of-month days for spot and futures (about 11/20 MB per day). Compute the upstream features exactly, then measure how well each Tier-B/C replacement tracks its upstream counterpart.
3. **Information test:** the rank IC of every feature against 1- to 60-bar forward returns and forward realized volatility, on the train split only. Compare with a shuffled copy of the same feature as the noise baseline. Drop any feature whose |IC| is not above the shuffled baseline. This is the same selection idea as EarnHFT's IC step, now restricted to train.
4. **Ablation:** train MacroHFT at 15m (about 40 min on the BOX) with (i) the current 36, (ii) the repaired set, and (iii) the repaired set without the book features. Only keep a feature group that improves validation.

## 5. Recommendation

- **For results comparable to the paper:** Tier A. It is the only route that reproduces the authors' inputs without approximation.
  - Tardis Solo Spot yearly (4 years back) is about $10.8k. Academic is $5.4k if eligible; an LLC probably isn't.
  - Crypto Lake's company plan ($500/mo, from 2022-11) is the cheapest full-depth option, but check the coverage and volume limits with their free sample first.
- **If paying is not an option:** Tier B, i.e. trade and train on the BTCUSDT perpetual for 2023-01 → now (~3.75 years).
  - Every input is then real, and the spread estimator is checked against Binance's own quotes.
  - Model funding payments in the reward. The current environment ignores them; they are negligible for 1m holds but not for multi-hour ones.
- **Either way:** run section 4 first on the free data (Tardis first-of-month days and the futures `bookTicker` window). It costs nothing, and it shows how much of the upstream book signal matters for 1m-1h BTC before you spend money.

## 6. Sources

- **Binance public data:**
  - https://data.binance.vision (S3 listings checked for `futures/um/daily/{bookDepth,bookTicker,aggTrades}` and `spot/daily`)
  - https://github.com/binance/binance-public-data (issues #6, #300, #334, #437, #447, #496; PR #348)
- **Tardis:**
  - https://docs.tardis.dev/downloadable-csv-files/data-types
  - https://docs.tardis.dev/downloadable-csv-files/api
  - https://docs.tardis.dev/faq/billing-and-subscriptions
  - The EarnHFT source: `EarnHFT/data_preprocess/download_code/download.py`
- **Other vendors:**
  - https://crypto-lake.com/coverage/
  - https://crypto-lake.com/subscribe/
  - https://www.coinapi.io/products/flat-files/pricing
- **Spread estimators:**
  - Ardia, Guidotti & Kroencke (2024) J. Financial Economics 161:103916, doi:10.1016/j.jfineco.2024.103916; `bidask` on PyPI (2.1.0) and CRAN; https://github.com/eguidotti/bidask
  - Corwin & Schultz (2012) J. Finance
  - Abdi & Ranaldo (2017) RFS
  - Roll (1984) J. Finance
  - Brauneis, Mestel, Riordan & Theissen (2021) J. Banking & Finance 124:106041
- **Depth, impact and flow:**
  - Kyle (1985) Econometrica
  - Amihud (2002) J. Financial Markets
  - López de Prado (2018) *Advances in Financial Machine Learning*, ch. 19
  - Easley, O'Hara, Yang & Zhang (2024) SSRN 4814346 (microstructure features on Binance 1-minute bars)
  - Easley, López de Prado & O'Hara (2012) VPIN; Andersen & Bondarenko (2014) J. Financial Markets 17
  - Cont, Kukanov & Stoikov (2014) J. Financial Econometrics 12(1)
  - Silantyev (2019) Digital Finance 1, doi:10.1007/s42521-019-00007-w
- **Order-book signals:**
  - Stoikov (2018) Quantitative Finance 18(12) (micro-price; needs L1 sizes)
  - Cartea, Donnelly & Jaimungal (2018) Applied Mathematical Finance 25(1) (volume imbalance)
  - Cao, Hansch & Wang (2009) J. Futures Markets (information in the depth profile)
