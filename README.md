# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xLonely1/gta6-trophy-rarity-forecast-2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xLonely1/gta6-trophy-rarity-forecast-2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts the earn rate of each of the 55 GTA VI PS5 trophies on Exophase at 2026-11-26T00:00:00Z, from the history of Exophase page reads in the case file. The model decomposes each trophy's earn rate into earners over players tracked, fits a saturating curve to each, and blends that with a logit-linear trend over the last 24 hours of reads and the last observed rate, then reports five quantiles (5/25/50/75/95) per trophy. The key insight it encodes is that Exophase rates fall when the player base jumps (new players have not earned the trophies yet) and rise as players play, so a naive flat forecast is badly wrong right after a refresh.

## Run it

```sh
python3 /bot/main.py
```

The program reads the case file from the input directory (`MAKEFUN_INPUT_DIR`, default `/input`) and writes one forecast file to the output directory (`MAKEFUN_OUTPUT_DIR`, default `/output`). Python 3.13, standard library only, no network.

---

<sub>Entered on [make.fun](https://make.fun/?ref=kkhbctt2&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
