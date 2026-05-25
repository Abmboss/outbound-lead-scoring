-- ============================================================
-- behavioral_features.sql
-- Base behavioral aggregations for lead scoring feature store.
-- Runs daily in BigQuery via Dataform/scheduled query.
--
-- Inputs : raw_events, dim_leads
-- Output : feat_behavioral (partitioned by score_date)
-- ============================================================

WITH

-- Anchor dates for rolling windows
params AS (
  SELECT
    CURRENT_DATE()                              AS ref_date,
    DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)   AS cutoff_7d,
    DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)  AS cutoff_30d,
    DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)  AS cutoff_90d
),

-- Filter active leads only
active_leads AS (
  SELECT lead_id
  FROM `{{project}}.{{dataset}}.dim_leads`
  WHERE
    status NOT IN ('DISQUALIFIED', 'CONVERTED', 'OPTED_OUT')
    AND partition_date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
),

-- Raw events within 90d window
events_90d AS (
  SELECT
    e.lead_id,
    e.event_date,
    e.event_type,
    e.channel,
    p.cutoff_7d,
    p.cutoff_30d,
    p.cutoff_90d,
    p.ref_date
  FROM `{{project}}.{{dataset}}.raw_events`         AS e
  CROSS JOIN params                                  AS p
  INNER JOIN active_leads                            AS l USING (lead_id)
  WHERE e.event_date BETWEEN p.cutoff_90d AND p.ref_date
),

-- Per-window aggregations (7d / 30d / 90d)
window_aggs AS (
  SELECT
    lead_id,

    -- 7-day window
    COUNTIF(event_date >= cutoff_7d)                                        AS w7d_n_events,
    COUNTIF(event_date >= cutoff_7d AND event_type = 'CALL_ANSWERED')       AS w7d_calls_answered,
    COUNTIF(event_date >= cutoff_7d AND event_type = 'CALL_MISSED')         AS w7d_calls_missed,
    COUNTIF(event_date >= cutoff_7d AND event_type = 'QUOTE_VIEWED')        AS w7d_quotes_viewed,
    COUNTIF(event_date >= cutoff_7d AND event_type = 'PROPOSAL_SENT')       AS w7d_proposals_sent,
    COUNTIF(event_date >= cutoff_7d AND event_type = 'EMAIL_OPENED')        AS w7d_email_opens,
    COUNTIF(event_date >= cutoff_7d AND event_type = 'EMAIL_CLICKED')       AS w7d_email_clicks,
    COUNT(DISTINCT IF(event_date >= cutoff_7d, channel, NULL))              AS w7d_channel_diversity,

    -- 30-day window
    COUNTIF(event_date >= cutoff_30d)                                       AS w30d_n_events,
    COUNTIF(event_date >= cutoff_30d AND event_type = 'CALL_ANSWERED')      AS w30d_calls_answered,
    COUNTIF(event_date >= cutoff_30d AND event_type = 'CALL_MISSED')        AS w30d_calls_missed,
    COUNTIF(event_date >= cutoff_30d AND event_type = 'QUOTE_VIEWED')       AS w30d_quotes_viewed,
    COUNTIF(event_date >= cutoff_30d AND event_type = 'PROPOSAL_SENT')      AS w30d_proposals_sent,
    COUNTIF(event_date >= cutoff_30d AND event_type = 'EMAIL_OPENED')       AS w30d_email_opens,
    COUNTIF(event_date >= cutoff_30d AND event_type = 'EMAIL_CLICKED')      AS w30d_email_clicks,
    COUNT(DISTINCT IF(event_date >= cutoff_30d, channel, NULL))             AS w30d_channel_diversity,

    -- 90-day window (all events in scope)
    COUNT(*)                                                                 AS w90d_n_events,
    COUNTIF(event_type = 'CALL_ANSWERED')                                   AS w90d_calls_answered,
    COUNTIF(event_type = 'CALL_MISSED')                                     AS w90d_calls_missed,
    COUNT(DISTINCT channel)                                                  AS w90d_channel_diversity,

    -- All-time derived metrics
    SAFE_DIVIDE(
      COUNTIF(event_type = 'CALL_ANSWERED'),
      COUNTIF(event_type IN ('CALL_ANSWERED', 'CALL_MISSED'))
    )                                                                        AS answer_rate_90d,

    MAX(IF(
      event_type IN ('CALL_ANSWERED', 'QUOTE_VIEWED', 'EMAIL_CLICKED', 'PORTAL_LOGIN'),
      event_date,
      NULL
    ))                                                                       AS last_engagement_date

  FROM events_90d
  GROUP BY lead_id
),

-- Final output with derived fields
final AS (
  SELECT
    l.lead_id,
    CURRENT_DATE()                                     AS score_date,

    -- Window aggregations
    w.w7d_n_events,
    w.w7d_calls_answered,
    w.w7d_calls_missed,
    w.w7d_quotes_viewed,
    w.w7d_proposals_sent,
    w.w7d_email_opens,
    w.w7d_email_clicks,
    w.w7d_channel_diversity,

    w.w30d_n_events,
    w.w30d_calls_answered,
    w.w30d_calls_missed,
    w.w30d_quotes_viewed,
    w.w30d_proposals_sent,
    w.w30d_email_opens,
    w.w30d_email_clicks,
    w.w30d_channel_diversity,

    w.w90d_n_events,
    w.w90d_calls_answered,
    w.w90d_calls_missed,
    w.w90d_channel_diversity,

    -- Derived engagement signals
    COALESCE(w.answer_rate_90d, 0)                     AS answer_rate_90d,
    DATE_DIFF(CURRENT_DATE(), w.last_engagement_date, DAY)
                                                       AS days_since_last_engagement,

    -- Engagement intensity score (0–100 normalized)
    ROUND(
      LEAST(
        100,
        COALESCE(w.w30d_n_events, 0) * 2
        + COALESCE(w.w30d_calls_answered, 0) * 5
        + COALESCE(w.w30d_quotes_viewed, 0) * 8
        + COALESCE(w.w30d_email_clicks, 0) * 4
        + COALESCE(w.w7d_n_events, 0) * 3   -- recency bonus
      ),
      2
    )                                                  AS engagement_score

  FROM active_leads AS l
  LEFT JOIN window_aggs AS w USING (lead_id)
)

SELECT * FROM final
