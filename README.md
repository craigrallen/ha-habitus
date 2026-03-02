# 🧠 Habitus

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-support-yellow?style=flat-square&logo=buy-me-a-coffee)](https://buymeacoffee.com/craigrallen)
[![HA Version](https://img.shields.io/badge/Home%20Assistant-2024.1%2B-blue?style=flat-square&logo=home-assistant)](https://www.home-assistant.io)
[![License](https://img.shields.io/github/license/craigrallen/ha-habitus?style=flat-square)](LICENSE)

**Behavioral intelligence for Home Assistant.**

Habitus learns the normal patterns of your home — energy use, temperatures, sensor activity — and tells you when something's off. No cloud. No accounts. Runs entirely on your HA device.

---

## What it does

- 📊 **Learns your home** — trains on years of long-term statistics already in your HA recorder
- 🔍 **Detects anomalies** — flags unusual patterns across all circuits, temperatures, and sensors
- 📡 **Publishes sensors** — `sensor.habitus_anomaly_score`, `binary_sensor.habitus_anomaly_detected`, and more
- 🌐 **Web UI** — built-in insights dashboard with hourly power baselines and run state
- 🔒 **100% local** — no data leaves your device, ever

---

## Installation

1. In Home Assistant: **Settings → Add-ons → Add-on Store → ⋮ → Repositories**
2. Add: `https://github.com/craigrallen/ha-habitus`
3. Find **Habitus** in the store and click **Install**
4. Start the add-on — it will appear in your sidebar automatically

---

## Sensors

| Entity | Description |
|--------|-------------|
| `sensor.habitus_anomaly_score` | 0–100 anomaly score (0 = normal, 100 = very unusual) |
| `binary_sensor.habitus_anomaly_detected` | `on` when score exceeds 70 |
| `sensor.habitus_training_days` | Days of history used to train the model |
| `sensor.habitus_entity_count` | Number of behavioral sensors being tracked |

---

## Configuration

```yaml
scan_interval_hours: 6   # How often to re-score (default: 6)
days_history: 365        # Training window in days (default: 365)
```

---

## How it works

On first run, Habitus queries your HA long-term statistics (no raw data is stored locally — HA is the source of truth). It builds a behavioral model using an Isolation Forest trained on hourly feature vectors: total power, average temperature, sensor activity, and time patterns.

Every subsequent run only fetches new data since the last run. The model is retrained periodically against the full window. Only the trained model weights and baseline statistics are stored (~100KB total).

---

## Automation example

```yaml
automation:
  - alias: "Habitus anomaly alert"
    trigger:
      - platform: state
        entity_id: binary_sensor.habitus_anomaly_detected
        to: "on"
    action:
      - service: notify.mobile_app
        data:
          title: "🧠 Habitus Alert"
          message: "Unusual behavior detected in your home."
```

---

## Support

If Habitus is useful to you, consider buying me a coffee ☕

[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/craigrallen)

---

## License

MIT
