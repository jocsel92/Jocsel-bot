# Jocsel-bot

Bot de trading Forex en MT5 con filtros de robustez para ejecución en vivo.

## Archivo principal

- `/home/runner/work/Jocsel-bot/Jocsel-bot/forex_bot.py`

## Mejoras implementadas

- Credenciales seguras por variables de entorno (sin secretos hardcodeados).
- Filtro de confianza ML con umbral dinámico por régimen de mercado.
- Uso efectivo de `min_pips` por símbolo antes de abrir operación.
- Alineación VWAP diaria/semanal/mensual con tolerancia configurable.
- Mejor manejo de errores y logging al modificar SL/TP (trailing y break-even).
- Bloqueo opcional alrededor de eventos macro vía `NEWS_EVENTS_UTC`.

## Variables de entorno requeridas

- `MT5_LOGIN`
- `MT5_PASSWORD`
- `MT5_SERVER`

## Variables opcionales

- `BOT_LOT` (por defecto `0.2`)
- `NEWS_EVENTS_UTC` (lista ISO UTC separada por comas)
- `NEWS_BLOCK_MINUTES` (por defecto `30`)
- `ORDER_FILLING_MODE` (`FOK` por defecto, o `IOC`)

## Ejecución

```bash
python /home/runner/work/Jocsel-bot/Jocsel-bot/forex_bot.py
```
