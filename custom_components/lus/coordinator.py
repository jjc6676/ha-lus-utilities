"""Imports LUS meter reads and estimated costs into long-term statistics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import logging

import aiohttp

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.models import (
    StatisticData,
    StatisticMeanType,
    StatisticMetaData,
)
from homeassistant.components.recorder.statistics import (
    async_add_external_statistics,
    get_last_statistics,
    statistics_during_period,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_PASSWORD, CONF_USERNAME, UnitOfEnergy, UnitOfVolume
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.aiohttp_client import async_create_clientsession
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import EnergyConverter, VolumeConverter

from .api import (
    FUEL_ELECTRICITY,
    FUEL_WASTE_WATER,
    FUEL_WATER,
    HourlyUsage,
    LusAuthError,
    LusClient,
    LusError,
    Meter,
    to_hourly,
)
from .analytics import Insights, build_insights, build_ledger
from .const import (
    CHUNK_DAYS,
    CONF_RECALCULATE_FROM,
    DEFAULT_BACKFILL,
    DOMAIN,
    INSIGHTS_HISTORY,
    RECALCULATE_LOOKBACK,
    UPDATE_INTERVAL,
)
from .cost import (
    SITE_TZ,
    BillEstimate,
    Rates,
    RateSchedule,
    bill_estimates,
    cycle_bounds,
    electricity_costs,
    garbage_cost,
    water_costs,
)

_LOGGER = logging.getLogger(__name__)

# fuel type -> (label, unit, unit class)
_FUELS = {
    FUEL_ELECTRICITY: ("electricity", UnitOfEnergy.KILO_WATT_HOUR, EnergyConverter.UNIT_CLASS),
    FUEL_WATER: ("water", UnitOfVolume.GALLONS, VolumeConverter.UNIT_CLASS),
    FUEL_WASTE_WATER: ("wastewater", UnitOfVolume.GALLONS, VolumeConverter.UNIT_CLASS),
}

# Fuels that get a cost statistic. Wastewater is billed from the water meter's
# gallons, so its charges are part of the water cost.
_COST_FUNCTIONS = {
    FUEL_ELECTRICITY: electricity_costs,
    FUEL_WATER: water_costs,
}

type LusConfigEntry = ConfigEntry[LusCoordinator]


@dataclass(frozen=True)
class LusData:
    """Estimated bills and spending insights for the sensors."""

    account_id: str
    current: BillEstimate
    previous: BillEstimate
    # None until there is usage to analyse, or if the analysis failed.
    insights: Insights | None = None


# (start of the last stored hour, its running sum); (None, 0.0) if none stored.
type LastStat = tuple[datetime | None, float]


def statistic_id(meter: Meter) -> str:
    """Statistic id for a meter's consumption, e.g. lus:1000000001_electricity_consumption."""
    label = _FUELS[meter.fuel_type][0]
    return f"{DOMAIN}:{meter.utility_account_id}_{label}_consumption"


def cost_statistic_id(meter: Meter) -> str:
    """Statistic id for a meter's estimated cost, e.g. lus:1000000001_electricity_cost."""
    label = _FUELS[meter.fuel_type][0]
    return f"{DOMAIN}:{meter.utility_account_id}_{label}_cost"


def total_cost_statistic_id(account_id: str) -> str:
    """Statistic id for the account's estimated total bill."""
    return f"{DOMAIN}:{account_id}_total_cost"


class LusCoordinator(DataUpdateCoordinator[LusData | None]):
    """Pulls new reads every few hours and appends them to statistics."""

    config_entry: LusConfigEntry

    def __init__(self, hass: HomeAssistant, entry: LusConfigEntry) -> None:
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=DOMAIN,
            update_interval=UPDATE_INTERVAL,
        )
        self.client = LusClient(
            async_create_clientsession(hass),
            entry.data[CONF_USERNAME],
            entry.data[CONF_PASSWORD],
        )
        # Hourly usage per meter for the spending insights, kept between
        # updates so stored statistics are only read once after start-up.
        self._history: dict[str, dict[datetime, float]] = {}

    async def _async_update_data(self) -> LusData | None:
        started_with = dict(self.config_entry.options)
        rates = RateSchedule.from_options(started_with)
        # After rates change, costs from this point on are recomputed and
        # overwrite what's stored; earlier hours are left as they are.
        recalculate_from = self._recalculate_from()
        try:
            await self.client.async_login()
            meters = [
                m for m in await self.client.async_get_meters() if m.fuel_type in _FUELS
            ]
            total_id = total_cost_statistic_id(self.client.account_id)
            total_last = await self._async_last(total_id, recalculate_from)
            costs = []
            usage: dict[Meter, list[HourlyUsage]] = {}
            for meter in meters:
                meter_costs = await self._async_import_meter(
                    meter, rates, total_last, recalculate_from, usage
                )
                if meter_costs is not None:
                    costs.append(meter_costs)
        except LusAuthError as err:
            raise ConfigEntryAuthFailed(str(err)) from err
        except (LusError, aiohttp.ClientError, TimeoutError) as err:
            raise UpdateFailed(str(err)) from err

        data = None
        if costs:
            totals = total_costs(costs, rates)
            self._add(
                total_id,
                total_last,
                totals,
                self._cost_metadata(total_id, f"LUS total cost {self.client.account_id}"),
            )
            current, previous = bill_estimates(
                totals, rates.billing_day, dt_util.now(SITE_TZ)
            )
            data = LusData(
                self.client.account_id,
                current,
                previous,
                await self._async_insights(usage, rates),
            )
        if recalculate_from is not None:
            _LOGGER.info("Recalculated costs from %s", recalculate_from)
            # Rates saved while this update ran ask for a recalculation of
            # their own. Clearing the request now would drop it, and writing
            # back the options read at the start would drop the rates too.
            if dict(self.config_entry.options) == started_with:
                options = {
                    k: v for k, v in started_with.items() if k != CONF_RECALCULATE_FROM
                }
                self.hass.config_entries.async_update_entry(
                    self.config_entry, options=options
                )
        return data

    def _recalculate_from(self) -> datetime | None:
        value = self.config_entry.options.get(CONF_RECALCULATE_FROM)
        if not value:
            return None
        day = date.fromisoformat(value)
        return datetime(day.year, day.month, day.day, tzinfo=SITE_TZ)

    async def _async_import_meter(
        self,
        meter: Meter,
        rates: Rates | RateSchedule,
        total_last: LastStat,
        recalculate_from: datetime | None = None,
        usage: dict[Meter, list[HourlyUsage]] | None = None,
    ) -> list[HourlyUsage] | None:
        """Import consumption and, for billed meters, cost. Returns the cost hours.

        A billed meter's usage hours are also put in `usage`, when given.
        """
        consumption_id = statistic_id(meter)
        cost_fn = _COST_FUNCTIONS.get(meter.fuel_type)
        cost_id = cost_statistic_id(meter) if cost_fn else None

        consumption_last = await self._async_last(consumption_id)
        today = dt_util.now(SITE_TZ).date()
        resume = [self._resume_date(consumption_last, meter, today, None)]
        if cost_id:
            cost_last = await self._async_last(cost_id, recalculate_from)
            # Cost depends on usage earlier in the billing cycle, so re-read
            # from the start of the cycle being resumed.
            resume.append(self._resume_date(cost_last, meter, today, rates.billing_day))
            # The bill sensors need the previous cycle's total cost too.
            resume.append(
                self._resume_date(
                    total_last, meter, today, rates.billing_day, previous_cycle=True
                )
            )
        fetch_from = min(resume)

        reads = []
        for start, end in _chunks(fetch_from, today, CHUNK_DAYS):
            reads.extend(await self.client.async_get_reads(meter, start, end))
        hours = to_hourly(
            reads, meter.reads_per_hour, stored_through=consumption_last[0]
        )

        label, unit, unit_class = _FUELS[meter.fuel_type]
        self._add(
            consumption_id,
            consumption_last,
            hours,
            StatisticMetaData(
                mean_type=StatisticMeanType.NONE,
                has_sum=True,
                name=f"LUS {label} {meter.utility_account_id}",
                source=DOMAIN,
                statistic_id=consumption_id,
                unit_class=unit_class,
                unit_of_measurement=unit,
            ),
        )
        if not cost_id:
            return None
        if usage is not None:
            usage[meter] = hours

        costs = cost_fn(hours, rates)
        self._add(
            cost_id,
            cost_last,
            costs,
            self._cost_metadata(cost_id, f"LUS {label} cost {meter.utility_account_id}"),
        )
        return costs

    async def _async_last(self, stat_id: str, before: datetime | None = None) -> LastStat:
        """The last stored hour, or the last one before `before` when given.

        With `before`, nothing stored earlier means everything is recomputed
        from the start of history.
        """
        recorder = get_instance(self.hass)
        if before is None:
            rows = await recorder.async_add_executor_job(
                get_last_statistics, self.hass, 1, stat_id, True, {"sum"}
            )
        else:
            rows = await recorder.async_add_executor_job(
                statistics_during_period,
                self.hass,
                before - RECALCULATE_LOOKBACK,
                before,
                {stat_id},
                "hour",
                None,
                {"sum"},
            )
            if not rows or not rows.get(stat_id):
                # Nothing in the weeks before the date: it is in the future,
                # or follows a long gap. The running sum still has to carry
                # on from whatever is stored earlier, however far back, or
                # the rewritten hours would restart from zero mid-history.
                rows = await recorder.async_add_executor_job(
                    statistics_during_period,
                    self.hass,
                    dt_util.utc_from_timestamp(0),
                    before - RECALCULATE_LOOKBACK,
                    {stat_id},
                    # Hourly, not a coarser period: those are widened to whole
                    # days or months and would take in hours after `before`.
                    "hour",
                    None,
                    {"sum"},
                )
        if not rows or not rows.get(stat_id):
            return None, 0.0
        row = rows[stat_id][-1]
        return dt_util.utc_from_timestamp(row["start"]), row["sum"] or 0.0

    async def _async_insights(
        self, usage: dict[Meter, list[HourlyUsage]], rates: Rates | RateSchedule
    ) -> Insights | None:
        """Analyse spending over the usage history.

        Insights are an extra on top of the statistics import, so a failure
        here is logged and leaves their sensors unknown instead of failing
        the update.
        """
        try:
            now = dt_util.now(SITE_TZ)
            oldest = now - INSIGHTS_HISTORY
            series = []
            for meter, hours in usage.items():
                history = self._history.get(meter.uuid)
                if history is None:
                    # First update since start-up: hours older than this
                    # update fetched come from the stored statistics.
                    before = hours[0].start if hours else now
                    history = await self._async_stored_usage(
                        statistic_id(meter), oldest, before
                    )
                    self._history[meter.uuid] = history
                history.update((h.start, h.value) for h in hours)
                for stale in [start for start in history if start < oldest]:
                    del history[stale]
                series.append(
                    (meter.fuel_type, [HourlyUsage(s, v) for s, v in history.items()])
                )
            return await self.hass.async_add_executor_job(
                _build_insights, series, rates, now
            )
        except Exception:  # noqa: BLE001 - must never break the import
            _LOGGER.exception("Could not work out LUS spending insights")
            return None

    async def _async_stored_usage(
        self, stat_id: str, start: datetime, end: datetime
    ) -> dict[datetime, float]:
        """Stored hourly usage for a statistic, by the start of each hour."""
        if end <= start:
            return {}
        rows = await get_instance(self.hass).async_add_executor_job(
            statistics_during_period,
            self.hass,
            start,
            end,
            {stat_id},
            "hour",
            {
                EnergyConverter.UNIT_CLASS: UnitOfEnergy.KILO_WATT_HOUR,
                VolumeConverter.UNIT_CLASS: UnitOfVolume.GALLONS,
            },
            {"state", "sum"},
        )
        usage: dict[datetime, float] = {}
        previous_sum: float | None = None
        for row in (rows or {}).get(stat_id, []):
            value = row.get("state")
            total = row.get("sum")
            if value is None and total is not None and previous_sum is not None:
                # Rows without a state still carry the running sum.
                value = total - previous_sum
            previous_sum = total if total is not None else previous_sum
            if value is not None:
                usage[dt_util.utc_from_timestamp(row["start"])] = value
        return usage

    @staticmethod
    def _resume_date(
        last: LastStat,
        meter: Meter,
        today: date,
        billing_day: int | None,
        previous_cycle: bool = False,
    ) -> date:
        """First day to fetch so everything after `last` can be computed.

        With a billing day, that's the start of the cycle `last` is in, or of
        the cycle before it when `previous_cycle` is set.
        """
        last_start, _ = last
        if last_start is None:
            return meter.first_read or (today - DEFAULT_BACKFILL)
        if billing_day is None:
            return last_start.astimezone(SITE_TZ).date()
        start = cycle_bounds(last_start, billing_day)[0]
        if previous_cycle:
            start = cycle_bounds(start - timedelta(seconds=1), billing_day)[0]
        return start.date()

    def _cost_metadata(self, stat_id: str, name: str) -> StatisticMetaData:
        return StatisticMetaData(
            mean_type=StatisticMeanType.NONE,
            has_sum=True,
            name=name,
            source=DOMAIN,
            statistic_id=stat_id,
            unit_class=None,
            unit_of_measurement=self.hass.config.currency,
        )

    def _add(
        self,
        stat_id: str,
        last: LastStat,
        hours: list[HourlyUsage],
        metadata: StatisticMetaData,
    ) -> None:
        """Append the hours after the last stored one, continuing its running sum."""
        last_start, total = last
        statistics = []
        for hour in hours:
            if last_start is not None and hour.start <= last_start:
                continue
            total += hour.value
            statistics.append(StatisticData(start=hour.start, state=hour.value, sum=total))
        if not statistics:
            _LOGGER.debug("No new complete hours for %s", stat_id)
            return
        _LOGGER.debug(
            "Adding %d hours to %s, through %s",
            len(statistics),
            stat_id,
            statistics[-1]["start"],
        )
        async_add_external_statistics(self.hass, metadata, statistics)


def total_costs(
    costs: list[list[HourlyUsage]], rates: Rates | RateSchedule
) -> list[HourlyUsage]:
    """Sum per-meter costs plus garbage, for hours every meter has reported."""
    by_meter = [{h.start: h.value for h in meter_costs} for meter_costs in costs]
    common = set(by_meter[0]).intersection(*by_meter[1:])
    return [
        HourlyUsage(hour, sum(m[hour] for m in by_meter) + garbage_cost(hour, rates))
        for hour in sorted(common)
    ]


def _build_insights(
    series: list[tuple[str, list[HourlyUsage]]],
    rates: Rates | RateSchedule,
    now: datetime,
) -> Insights:
    """Cost out the usage history and analyse it; run off the event loop."""
    return build_insights(build_ledger(series, rates), rates, now)


def _chunks(start: date, end: date, days: int) -> list[tuple[date, date]]:
    """Split an inclusive date range into inclusive chunks of at most `days` days."""
    chunks = []
    while start <= end:
        chunk_end = min(start + timedelta(days=days - 1), end)
        chunks.append((start, chunk_end))
        start = chunk_end + timedelta(days=1)
    return chunks
