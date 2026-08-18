-- Observed demand
CREATE TABLE demand_daily (
    date        DATE PRIMARY KEY,
    demand      DOUBLE PRECISION NOT NULL,
    is_imputed  BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Weather forecasts AS ISSUED
CREATE TABLE weather_forecast (
    target_date     DATE PRIMARY KEY,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    dew             DOUBLE PRECISION,
    precip          DOUBLE PRECISION,
    snow            DOUBLE PRECISION,
    tempmax         DOUBLE PRECISION,
    tempmin         DOUBLE PRECISION,
    humidity        DOUBLE PRECISION,
    snowdepth       DOUBLE PRECISION,
    solarenergy     DOUBLE PRECISION,
    cloudcover      DOUBLE PRECISION
);

-- Observed weather
CREATE TABLE weather_daily (
    date         DATE PRIMARY KEY,
    dew          DOUBLE PRECISION,
    precip       DOUBLE PRECISION,
    snow         DOUBLE PRECISION,
    tempmax      DOUBLE PRECISION,
    tempmin      DOUBLE PRECISION,
    humidity     DOUBLE PRECISION,
    snowdepth    DOUBLE PRECISION,
    solarenergy  DOUBLE PRECISION,
    cloudcover   DOUBLE PRECISION,
    sunrise      DOUBLE PRECISION,
    sunset       DOUBLE PRECISION
);

-- One row per prediction
CREATE TABLE predictions (
    target_date         DATE PRIMARY KEY,
    predicted_demand    DOUBLE PRECISION NOT NULL,
    model_version       TEXT NOT NULL,
    features            JSONB NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per scored prediction. Recent rows are refreshed because EIA can revise
-- recently published demand values.
CREATE TABLE prediction_scores (
    target_date         DATE PRIMARY KEY REFERENCES predictions(target_date),
    predicted_demand    DOUBLE PRECISION NOT NULL,
    actual_demand       DOUBLE PRECISION NOT NULL,
    error               DOUBLE PRECISION NOT NULL,
    abs_error           DOUBLE PRECISION NOT NULL,
    pct_error           DOUBLE PRECISION NOT NULL,
    abs_pct_error       DOUBLE PRECISION NOT NULL,
    model_version       TEXT NOT NULL,
    is_imputed          BOOLEAN NOT NULL DEFAULT FALSE,
    scored_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
