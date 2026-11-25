ALTER TABLE demand_daily
    ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'eia'
    CHECK (source IN ('eia', 'nyiso'));
