"use client";
import { useState, useEffect } from "react";

const ML_URL = process.env.NEXT_PUBLIC_ML_URL || "http://localhost:8000";

const AMENITIES = [
  { key: "amenity_gym", label: "Gym", icon: "🏋️" },
  { key: "amenity_security", label: "24×7 Security", icon: "🔒" },
  { key: "amenity_clubhouse", label: "Clubhouse", icon: "🏛️" },
  { key: "amenity_pool", label: "Swimming Pool", icon: "🏊" },
  { key: "amenity_gardens", label: "Gardens", icon: "🌳" },
  { key: "amenity_kids_play", label: "Kids Play Area", icon: "🛝" },
  { key: "amenity_indoor_games", label: "Indoor Games", icon: "🎮" },
  { key: "amenity_jogging", label: "Jogging Track", icon: "🏃" },
  { key: "amenity_intercom", label: "Intercom", icon: "📞" },
  { key: "amenity_maintenance", label: "Maintenance Staff", icon: "🛠️" },
  { key: "amenity_gas", label: "Gas Connection", icon: "🔥" },
];

export default function Home() {
  const [locations, setLocations] = useState([]);
  const [serviceStatus, setServiceStatus] = useState(null);
  const [showAmenities, setShowAmenities] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [feedbackPrice, setFeedbackPrice] = useState("");

  // New feature states
  const [emiData, setEmiData] = useState(null);
  const [emiTenure, setEmiTenure] = useState(20);
  const [emiRate, setEmiRate] = useState(8.5);
  const [emiDownPayment, setEmiDownPayment] = useState(20);
  const [similarProps, setSimilarProps] = useState([]);
  const [neighborhoodStats, setNeighborhoodStats] = useState(null);
  const [topNeighborhoods, setTopNeighborhoods] = useState([]);
  const [showCompare, setShowCompare] = useState(false);
  const [compareLocations, setCompareLocations] = useState([]);
  const [compareData, setCompareData] = useState([]);
  const [explanation, setExplanation] = useState(null);
  const [rentBuyData, setRentBuyData] = useState(null);
  const [priceTrend, setPriceTrend] = useState(null);
  const [rentBuyHorizon, setRentBuyHorizon] = useState(10);

  const [form, setForm] = useState({
    location: "",
    bhk: 2,
    area_sqft: 800,
    is_resale: 0,
    lift: 1,
    parking: 1,
    actual_listed_price: "",
    ...Object.fromEntries(AMENITIES.map((a) => [a.key, 0])),
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [error, setError] = useState("");
  const [bootState, setBootState] = useState("checking"); // "checking" | "waking" | "ready" | "down"
  const [bootSeconds, setBootSeconds] = useState(0);

  useEffect(() => {
    let attempts = 0;
    let intervalId = null;
    let secondsId = null;

    const tryBoot = async () => {
      attempts += 1;
      try {
        const r = await fetch(`${ML_URL}/`, { cache: "no-store" });
        if (!r.ok) throw new Error("not ok");
        const d = await r.json();
        setServiceStatus(d);
        setBootState("ready");
        if (intervalId) clearInterval(intervalId);
        if (secondsId) clearInterval(secondsId);
        // load locations once boot succeeds
        fetch(`${ML_URL}/locations`).then((r) => r.json()).then((d) => {
          const locs = d.locations || [];
          setLocations(locs);
          if (locs.length > 0) setForm((f) => ({ ...f, location: locs[0] }));
        }).catch(() => setLocations([]));
      } catch (e) {
        // First failure usually means free-tier server is waking up
        if (attempts >= 2) setBootState("waking");
        if (attempts > 30) {
          setBootState("down");
          setServiceStatus({ status: "offline" });
          if (intervalId) clearInterval(intervalId);
          if (secondsId) clearInterval(secondsId);
        }
      }
    };

    tryBoot();
    intervalId = setInterval(tryBoot, 3000); // retry every 3s
    secondsId = setInterval(() => setBootSeconds((s) => s + 1), 1000);

    return () => {
      if (intervalId) clearInterval(intervalId);
      if (secondsId) clearInterval(secondsId);
    };
  }, []);

  // Animated loading steps
  useEffect(() => {
    if (!loading) return;
    const steps = [0, 1, 2, 3];
    let i = 0;
    const interval = setInterval(() => {
      setLoadingStep(steps[i % steps.length]);
      i++;
    }, 600);
    return () => clearInterval(interval);
  }, [loading]);

  const analyze = async () => {
    if (!form.location) {
      setError("Please select a location");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    setFeedbackSent(false);
    setShowFeedback(false);

    try {
      const payload = { ...form };
      if (payload.actual_listed_price) {
        payload.actual_listed_price = +payload.actual_listed_price;
      } else {
        delete payload.actual_listed_price;
      }

      const res = await fetch(`${ML_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `Service returned ${res.status}`);
      }
      const data = await res.json();
      setResult(data);

      // Fetch extra features in parallel
      const predicted = data.price_prediction.predicted_price;
      const loanAmount = Math.round(predicted * (1 - emiDownPayment / 100));

      Promise.all([
        fetch(`${ML_URL}/similar/${form.location}/${form.bhk}/${form.area_sqft}`).then(r => r.json()).catch(() => ({ similar_properties: [] })),
        fetch(`${ML_URL}/neighborhood/${form.location}`).then(r => r.json()).catch(() => null),
        fetch(`${ML_URL}/emi`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ principal: loanAmount, annual_rate: emiRate, tenure_years: emiTenure }),
        }).then(r => r.json()).catch(() => null),
        fetch(`${ML_URL}/top-neighborhoods?n=8&sort=expensive`).then(r => r.json()).catch(() => ({ neighborhoods: [] })),
        fetch(`${ML_URL}/explain`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(form),
        }).then(r => r.ok ? r.json() : null).catch(() => null),
        fetch(`${ML_URL}/rent-vs-buy`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            property_price: predicted, bhk: form.bhk, area_sqft: form.area_sqft,
            horizon_years: rentBuyHorizon, down_payment_pct: emiDownPayment,
            loan_rate: emiRate, loan_tenure: emiTenure,
          }),
        }).then(r => r.ok ? r.json() : null).catch(() => null),
        fetch(`${ML_URL}/price-trend/${form.location}`).then(r => r.ok ? r.json() : null).catch(() => null),
      ]).then(([similar, nbhd, emi, topN, explain, rentBuy, trend]) => {
        setSimilarProps(similar.similar_properties || []);
        setNeighborhoodStats(nbhd);
        setEmiData(emi);
        setTopNeighborhoods(topN.neighborhoods || []);
        setExplanation(explain);
        setRentBuyData(rentBuy);
        setPriceTrend(trend);
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async () => {
    if (!result?.prediction_id) return;
    try {
      await fetch(`${ML_URL}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prediction_id: result.prediction_id,
          actual_price: feedbackPrice ? +feedbackPrice : null,
        }),
      });
      setFeedbackSent(true);
    } catch (e) {
      console.error(e);
    }
  };

  const toggleAmenity = (key) => setForm((f) => ({ ...f, [key]: f[key] ? 0 : 1 }));

  const recalcEMI = async (rate, tenure, downPct) => {
    if (!result) return;
    const predicted = result.price_prediction.predicted_price;
    const loanAmount = Math.round(predicted * (1 - downPct / 100));
    try {
      const res = await fetch(`${ML_URL}/emi`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ principal: loanAmount, annual_rate: rate, tenure_years: tenure }),
      });
      const data = await res.json();
      setEmiData(data);
    } catch (e) {
      console.error(e);
    }
  };

  const runCompare = async () => {
    if (compareLocations.length < 2) return;
    try {
      const res = await fetch(`${ML_URL}/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ locations: compareLocations }),
      });
      const data = await res.json();
      setCompareData(data.comparison || []);
    } catch (e) {
      console.error(e);
    }
  };

  const formatINR = (num) => {
    if (!num) return "—";
    if (num >= 1e7) return `₹${(num / 1e7).toFixed(2)} Cr`;
    if (num >= 1e5) return `₹${(num / 1e5).toFixed(2)} L`;
    return `₹${num.toLocaleString()}`;
  };

  const totalAmenities = AMENITIES.reduce((sum, a) => sum + form[a.key], 0);

  const verdictColor = (verdict) => {
    if (!verdict) return "#94a3b8";
    if (verdict.includes("Excellent")) return "#10b981";
    if (verdict.includes("Good")) return "#22d3ee";
    if (verdict.includes("Fair")) return "#fbbf24";
    return "#ef4444";
  };

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: globalCSS }} />

      {/* COLD-START OVERLAY: shown only on first visit when free-tier server is waking up */}
      {bootState === "waking" && (
        <div className="cold-overlay">
          <div className="cold-card">
            <div className="cold-spinner" />
            <h2>Waking up the AI...</h2>
            <p>
              PropWise runs on a free-tier server that sleeps when no one is using it.
              First visit takes ~30-60 seconds to wake up. You'll only see this once.
            </p>
            <div className="cold-progress">
              <div className="cold-progress-bar" style={{ width: `${Math.min(bootSeconds * 1.7, 95)}%` }} />
            </div>
            <p className="cold-secs">{bootSeconds}s elapsed</p>
          </div>
        </div>
      )}

      {bootState === "down" && (
        <div className="cold-overlay">
          <div className="cold-card">
            <h2>⚠️ Service unavailable</h2>
            <p>
              The ML backend isn't responding. This usually means the free-tier server is being redeployed.
              Please refresh in a minute.
            </p>
            <button className="btn btn-primary" onClick={() => location.reload()}>Try again</button>
          </div>
        </div>
      )}

      <div className="bg-gradient">
        <div className="bg-orbs">
          <div className="orb orb-1" />
          <div className="orb orb-2" />
          <div className="orb orb-3" />
        </div>

        <div className="container">
          {/* HEADER */}
          <header className="header">
            <div className="brand">
              <div className="brand-icon">🏠</div>
              <div>
                <div className="brand-name">PropWise</div>
                <div className="brand-tagline">AI Real Estate Intelligence</div>
              </div>
            </div>
            {serviceStatus && (
              <div className="status-pill">
                <span className={`status-dot ${serviceStatus.status === "ready" ? "online" : "offline"}`} />
                <span>{serviceStatus.status === "ready" ? "Models Online" : "Offline"}</span>
              </div>
            )}
          </header>

          {/* HERO */}
          <section className="hero">
            <h1 className="hero-title">
              Mumbai's smartest <span className="gradient-text">property AI</span>
            </h1>
            <p className="hero-subtitle">
              Three ML models. Real Mumbai data. Instant analysis.
            </p>
            {serviceStatus?.models && (
              <div className="model-badges">
                <ModelBadge label="Price" value={`${serviceStatus.models.price_predictor.accuracy.toFixed(0)}%`} loaded={serviceStatus.models.price_predictor.loaded} />
                <ModelBadge label="Investment" value={`${serviceStatus.models.investment_score.accuracy.toFixed(0)}%`} loaded={serviceStatus.models.investment_score.loaded} />
                <ModelBadge label="Anomaly" value={`${serviceStatus.models.anomaly_detector.anomaly_rate.toFixed(0)}%`} loaded={serviceStatus.models.anomaly_detector.loaded} />
                <ModelBadge label="Trained on" value={`${(serviceStatus.models.price_predictor.trained_on || 0).toLocaleString()}`} loaded={true} />
              </div>
            )}
          </section>

          {/* INPUT CARD */}
          <section className="input-card glass-card">
            <h2 className="section-title">Property Details</h2>

            <div className="form-grid">
              <Field label="Location" hint="Mumbai neighborhood">
                <select className="input" value={form.location} onChange={(e) => setForm({ ...form, location: e.target.value })}>
                  <option value="">— Select neighborhood —</option>
                  {locations.map((loc) => (
                    <option key={loc} value={loc}>{loc.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}</option>
                  ))}
                </select>
              </Field>

              <Field label="Bedrooms (BHK)">
                <div className="pill-group">
                  {[1, 2, 3, 4, 5].map((n) => (
                    <button key={n} className={`pill ${form.bhk === n ? "active" : ""}`} onClick={() => setForm({ ...form, bhk: n })}>
                      {n} BHK
                    </button>
                  ))}
                </div>
              </Field>

              <Field label="Carpet Area" hint="In square feet">
                <div className="number-input">
                  <input type="number" min="100" max="20000" className="input" value={form.area_sqft} onChange={(e) => setForm({ ...form, area_sqft: +e.target.value })} />
                  <span className="suffix">sqft</span>
                </div>
              </Field>

              <Field label="Property Type">
                <div className="pill-group">
                  <button className={`pill ${form.is_resale === 0 ? "active" : ""}`} onClick={() => setForm({ ...form, is_resale: 0 })}>New / Under Construction</button>
                  <button className={`pill ${form.is_resale === 1 ? "active" : ""}`} onClick={() => setForm({ ...form, is_resale: 1 })}>Resale</button>
                </div>
              </Field>

              <Field label="Building Features">
                <div className="checkbox-row">
                  <Checkbox label="Lift" checked={form.lift} onChange={(v) => setForm({ ...form, lift: v })} />
                  <Checkbox label="Parking" checked={form.parking} onChange={(v) => setForm({ ...form, parking: v })} />
                </div>
              </Field>

              <Field label="Listed Price (Optional)" hint="To compare against fair value">
                <div className="number-input">
                  <span className="prefix">₹</span>
                  <input type="number" placeholder="e.g. 9500000" className="input" value={form.actual_listed_price} onChange={(e) => setForm({ ...form, actual_listed_price: e.target.value })} />
                </div>
              </Field>
            </div>

            {/* Amenities Toggle */}
            <div className="amenities-section">
              <button className={`amenities-toggle ${showAmenities ? "open" : ""}`} onClick={() => setShowAmenities(!showAmenities)}>
                <span className="chevron">▶</span>
                <span>Premium Amenities</span>
                <span className="amenities-count">{totalAmenities} of {AMENITIES.length} selected</span>
              </button>
              {showAmenities && (
                <div className="amenities-grid">
                  {AMENITIES.map((a) => (
                    <button
                      key={a.key}
                      className={`amenity-chip ${form[a.key] === 1 ? "active" : ""}`}
                      onClick={() => toggleAmenity(a.key)}
                    >
                      <span className="amenity-icon">{a.icon}</span>
                      <span>{a.label}</span>
                      {form[a.key] === 1 && <span className="check">✓</span>}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <button className={`analyze-btn ${loading || !form.location ? "disabled" : ""}`} onClick={analyze} disabled={loading || !form.location}>
              {loading ? "Analyzing..." : "Run AI Analysis →"}
            </button>

            {error && <div className="error-box">⚠ {error}</div>}
          </section>

          {/* LOADING ANIMATION */}
          {loading && (
            <section className="glass-card loading-card">
              <div className="loading-models">
                <LoadingStep label="Price Predictor" active={loadingStep >= 0} done={loadingStep > 0} />
                <LoadingStep label="Investment Analysis" active={loadingStep >= 1} done={loadingStep > 1} />
                <LoadingStep label="Anomaly Detection" active={loadingStep >= 2} done={loadingStep > 2} />
                <LoadingStep label="Generating Insights" active={loadingStep >= 3} done={loadingStep > 3} />
              </div>
            </section>
          )}

          {/* RESULTS */}
          {result && (
            <section className="results">
              {/* PRICE HERO */}
              <div className="glass-card price-hero">
                <div className="price-hero-content">
                  <div className="price-label">PREDICTED FAIR PRICE</div>
                  <div className="price-value gradient-text">{formatINR(result.price_prediction.predicted_price)}</div>
                  <div className="price-range">
                    Range: {formatINR(result.price_prediction.price_range_low)} – {formatINR(result.price_prediction.price_range_high)}
                  </div>
                  <div className="price-sub">
                    ₹{result.price_prediction.price_per_sqft.toLocaleString()}/sqft · {result.price_prediction.model_accuracy}% model accuracy
                  </div>
                </div>
                <div className="price-meter">
                  <div className="meter-circle">
                    <svg viewBox="0 0 100 100" className="meter-svg">
                      <circle cx="50" cy="50" r="42" className="meter-bg" />
                      <circle cx="50" cy="50" r="42" className="meter-fill" style={{ strokeDasharray: `${result.price_prediction.model_accuracy * 2.64} 264` }} />
                    </svg>
                    <div className="meter-text">
                      <div className="meter-value">{result.price_prediction.model_accuracy}%</div>
                      <div className="meter-label">accuracy</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* NEGOTIATION INSIGHT */}
              {result.negotiation && (
                <div className={`glass-card negotiation-card ${result.negotiation.difference_pct > 5 ? "warning" : result.negotiation.difference_pct < -5 ? "alert" : "success"}`}>
                  <div className="neg-header">💰 Listing Comparison</div>
                  <div className="neg-grid">
                    <div>
                      <div className="neg-label">Listed Price</div>
                      <div className="neg-value">{formatINR(result.negotiation.listed_price)}</div>
                    </div>
                    <div className="neg-arrow">→</div>
                    <div>
                      <div className="neg-label">Fair Value</div>
                      <div className="neg-value">{formatINR(result.negotiation.predicted_fair_price)}</div>
                    </div>
                    <div className={`neg-diff ${result.negotiation.difference_pct > 0 ? "over" : "under"}`}>
                      {result.negotiation.difference_pct > 0 ? "+" : ""}{result.negotiation.difference_pct}%
                    </div>
                  </div>
                  <div className="neg-verdict">{result.negotiation.verdict}</div>
                </div>
              )}

              {/* MODEL ANALYSIS GRID */}
              <div className="analysis-grid">
                {/* INVESTMENT SCORE */}
                {result.investment_analysis && (
                  <div className="glass-card model-card">
                    <div className="model-header">
                      <span className="model-icon">📊</span>
                      <span className="model-title">Investment Analysis</span>
                    </div>
                    <div className="investment-score">
                      <div className="score-circle" style={{ background: `conic-gradient(${verdictColor(result.investment_analysis.verdict)} ${result.investment_analysis.score * 3.6}deg, rgba(255,255,255,0.05) 0deg)` }}>
                        <div className="score-inner">
                          <div className="score-num">{result.investment_analysis.score}</div>
                          <div className="score-of">/100</div>
                        </div>
                      </div>
                      <div>
                        <div className="verdict" style={{ color: verdictColor(result.investment_analysis.verdict) }}>{result.investment_analysis.verdict}</div>
                        <div className="confidence">{result.investment_analysis.confidence}% model confidence</div>
                      </div>
                    </div>
                    <div className="rentbuy-hint" style={{ marginTop: -10, marginBottom: 12, fontSize: 11 }}>
                      ℹ️ Score compares predicted fair price to area median. If you entered a listed price, the negotiation panel below uses real market data.
                    </div>
                    <div className="prob-bars">
                      {Object.entries(result.investment_analysis.probabilities).map(([name, pct]) => (
                        <div key={name} className="prob-row">
                          <span className="prob-label">{name}</span>
                          <div className="prob-bar">
                            <div className="prob-fill" style={{ width: `${pct}%`, background: verdictColor(name) }} />
                          </div>
                          <span className="prob-pct">{pct.toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* ANOMALY DETECTION */}
                {result.anomaly_check && (
                  <div className={`glass-card model-card ${result.anomaly_check.is_anomalous ? "anomaly" : "normal"}`}>
                    <div className="model-header">
                      <span className="model-icon">{result.anomaly_check.is_anomalous ? "🚨" : "✅"}</span>
                      <span className="model-title">Anomaly Detection</span>
                    </div>
                    <div className="anomaly-badge">
                      {result.anomaly_check.is_anomalous ? "SUSPICIOUS" : "VERIFIED NORMAL"}
                    </div>
                    <div className="anomaly-score">
                      Score: <strong>{result.anomaly_check.score}</strong> (negative = anomaly)
                    </div>
                    <div className="anomaly-verdict">{result.anomaly_check.verdict}</div>
                  </div>
                )}
              </div>

              {/* INSIGHTS */}
              <div className="glass-card insights-card">
                <div className="model-header">
                  <span className="model-icon">💡</span>
                  <span className="model-title">Smart Insights</span>
                </div>
                <div className="insight-tags">
                  {result.location_known ? (
                    <Tag color="#10b981">✓ High-confidence area</Tag>
                  ) : (
                    <Tag color="#fbbf24">⚠ Estimated from similar areas</Tag>
                  )}
                  <Tag color="#3b82f6">{result.insights.amenity_count} amenities</Tag>
                  <Tag color="#8b5cf6">{result.insights.model_confidence} confidence</Tag>
                </div>
              </div>

              {/* FEEDBACK */}
              <div className="glass-card feedback-card">
                <div className="feedback-header">
                  <span className="model-icon">🎯</span>
                  <span className="model-title">Help PropWise learn</span>
                </div>
                {feedbackSent ? (
                  <div className="feedback-thanks">✅ Thanks! Your input helps train future model versions.</div>
                ) : !showFeedback ? (
                  <button className="feedback-cta" onClick={() => setShowFeedback(true)}>
                    Did you actually buy/see this property? Tell us the real price →
                  </button>
                ) : (
                  <div className="feedback-form">
                    <input
                      type="number"
                      placeholder="Actual price (₹)"
                      className="input"
                      value={feedbackPrice}
                      onChange={(e) => setFeedbackPrice(e.target.value)}
                    />
                    <button className="feedback-submit" onClick={submitFeedback}>Submit</button>
                  </div>
                )}
              </div>

              {/* ── SHAP EXPLAINABILITY ── */}
              {explanation && (
                <div className="glass-card explain-card">
                  <div className="model-header">
                    <span className="model-icon">🔍</span>
                    <span className="model-title">Why this price? (Model Explainability)</span>
                  </div>
                  <p className="explain-intro">
                    Base price for any Mumbai property: <strong>{formatINR(explanation.base_price)}</strong>. Here's how this property's features adjusted that:
                  </p>
                  <div className="explain-cols">
                    <div className="explain-col">
                      <div className="explain-col-title up">↑ Increases price</div>
                      {explanation.top_increases.map((c, i) => (
                        <div key={i} className="explain-row">
                          <span className="explain-feat">{c.pretty_name}</span>
                          <span className="explain-impact up">+{c.price_impact_pct}%</span>
                        </div>
                      ))}
                      {explanation.top_increases.length === 0 && <div className="explain-empty">No major positive factors</div>}
                    </div>
                    <div className="explain-col">
                      <div className="explain-col-title down">↓ Decreases price</div>
                      {explanation.top_decreases.map((c, i) => (
                        <div key={i} className="explain-row">
                          <span className="explain-feat">{c.pretty_name}</span>
                          <span className="explain-impact down">{c.price_impact_pct}%</span>
                        </div>
                      ))}
                      {explanation.top_decreases.length === 0 && <div className="explain-empty">No major negative factors</div>}
                    </div>
                  </div>
                </div>
              )}

              {/* ── RENT VS BUY ── */}
              {rentBuyData && (
                <div className="glass-card rentbuy-card">
                  <div className="model-header">
                    <span className="model-icon">⚖️</span>
                    <span className="model-title">Rent vs Buy Analysis</span>
                  </div>
                  <div className="rentbuy-verdict" style={{
                    background: rentBuyData.recommendation === "buy" ? "rgba(16,185,129,0.1)" : "rgba(251,191,36,0.1)",
                    borderColor: rentBuyData.recommendation === "buy" ? "rgba(16,185,129,0.3)" : "rgba(251,191,36,0.3)",
                  }}>
                    <span className="rentbuy-icon">{rentBuyData.recommendation === "buy" ? "🏠" : "🔑"}</span>
                    <span style={{ color: rentBuyData.recommendation === "buy" ? "#6ee7b7" : "#fde68a" }}>{rentBuyData.verdict}</span>
                  </div>
                  <div className="rentbuy-grid">
                    <div className="rentbuy-col">
                      <div className="rentbuy-title">🏠 BUYING</div>
                      <div className="rentbuy-row"><span>Down payment:</span> <strong>{formatINR(rentBuyData.buying.down_payment)}</strong></div>
                      <div className="rentbuy-row"><span>Monthly EMI:</span> <strong>{formatINR(rentBuyData.buying.monthly_emi)}</strong></div>
                      <div className="rentbuy-row"><span>Total interest:</span> <strong>{formatINR(rentBuyData.buying.total_interest)}</strong></div>
                      <div className="rentbuy-row"><span>Maintenance:</span> <strong>{formatINR(rentBuyData.buying.total_maintenance)}</strong></div>
                      <div className="rentbuy-row"><span>Property value in {rentBuyData.scenario_horizon_years}y:</span> <strong>{formatINR(rentBuyData.buying.future_property_value)}</strong></div>
                      <div className="rentbuy-row total">
                        <span>{rentBuyData.buying.net_cost < 0 ? "NET GAIN:" : "NET COST:"}</span>
                        <strong style={{color: rentBuyData.buying.net_cost < 0 ? "#6ee7b7" : undefined}}>
                          {rentBuyData.buying.net_cost < 0 ? "+" : ""}{formatINR(Math.abs(rentBuyData.buying.net_cost))}
                        </strong>
                      </div>
                    </div>
                    <div className="rentbuy-col">
                      <div className="rentbuy-title">🔑 RENTING</div>
                      <div className="rentbuy-row"><span>Starting rent:</span> <strong>{formatINR(rentBuyData.renting.starting_monthly_rent)}/mo</strong></div>
                      <div className="rentbuy-row"><span>Total rent ({rentBuyData.scenario_horizon_years}y):</span> <strong>{formatINR(rentBuyData.renting.total_rent_paid)}</strong></div>
                      <div className="rentbuy-row"><span>Down payment invested:</span> <strong>{formatINR(rentBuyData.renting.down_payment_invested_value)}</strong></div>
                      <div className="rentbuy-row"><span>Investment gain:</span> <strong style={{color: "#6ee7b7"}}>+{formatINR(rentBuyData.renting.investment_gain)}</strong></div>
                      <div className="rentbuy-row total">
                        <span>{rentBuyData.renting.net_cost < 0 ? "NET GAIN:" : "NET COST:"}</span>
                        <strong style={{color: rentBuyData.renting.net_cost < 0 ? "#6ee7b7" : undefined}}>
                          {rentBuyData.renting.net_cost < 0 ? "+" : ""}{formatINR(Math.abs(rentBuyData.renting.net_cost))}
                        </strong>
                      </div>
                    </div>
                  </div>
                  {rentBuyData.breakeven_year && (
                    <div className="rentbuy-breakeven">
                      💡 Break-even point: <strong>Year {rentBuyData.breakeven_year}</strong> — buying becomes cheaper after this.
                    </div>
                  )}
                  {rentBuyData.rent_estimate.adjusted && (
                    <div className="rentbuy-note" style={{ marginTop: 10, padding: "8px 12px", background: "rgba(251,191,36,0.08)", border: "1px solid rgba(251,191,36,0.25)", borderRadius: 8, fontSize: 11, color: "#fde68a" }}>
                      ⚠️ {rentBuyData.rent_estimate.note}
                    </div>
                  )}
                  <div className="rentbuy-hint">
                    Based on {rentBuyData.rent_estimate.based_on_listings} similar Mumbai rentals · Assumes {rentBuyData.assumptions.property_appreciation_pct}% property appreciation, {rentBuyData.assumptions.alternate_investment_return}% return on alternative investment
                  </div>
                </div>
              )}

              {/* ── PRICE TREND ── */}
              {priceTrend && (
                <div className="glass-card">
                  <div className="model-header">
                    <span className="model-icon">📈</span>
                    <span className="model-title">{priceTrend.location.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())} Price Trend (5-year)</span>
                  </div>
                  <div className="trend-bars">
                    {priceTrend.trend.map((t, i) => {
                      const max = Math.max(...priceTrend.trend.map(x => x.price_per_sqft));
                      const pct = (t.price_per_sqft / max) * 100;
                      return (
                        <div key={i} className="trend-bar-row">
                          <span className="trend-year">{t.year}</span>
                          <div className="trend-bar-track">
                            <div className="trend-bar-fill" style={{ width: `${pct}%` }} />
                          </div>
                          <span className="trend-value">₹{t.price_per_sqft.toLocaleString()}/sqft</span>
                        </div>
                      );
                    })}
                  </div>
                  <div className="trend-footer">
                    📊 5-year growth: <strong>+{priceTrend.five_year_growth_pct}%</strong> (Mumbai market average)
                  </div>
                  <div className="rentbuy-hint" style={{ marginTop: 8 }}>
                    ℹ️ Projected using Mumbai's ~8% CAGR. Not based on historical listing data — directional only.
                  </div>
                </div>
              )}

              {/* ── EMI CALCULATOR ── */}
              {emiData && (
                <div className="glass-card emi-card">
                  <div className="model-header">
                    <span className="model-icon">🏦</span>
                    <span className="model-title">EMI Calculator</span>
                  </div>
                  <div className="emi-controls">
                    <div className="emi-slider-row">
                      <label>Down Payment: <strong>{emiDownPayment}%</strong> ({formatINR(result.price_prediction.predicted_price * emiDownPayment / 100)})</label>
                      <input type="range" min="10" max="50" step="5" value={emiDownPayment}
                        onChange={(e) => { setEmiDownPayment(+e.target.value); recalcEMI(emiRate, emiTenure, +e.target.value); }} />
                    </div>
                    <div className="emi-slider-row">
                      <label>Interest Rate: <strong>{emiRate}%</strong></label>
                      <input type="range" min="6.5" max="12" step="0.25" value={emiRate}
                        onChange={(e) => { setEmiRate(+e.target.value); recalcEMI(+e.target.value, emiTenure, emiDownPayment); }} />
                    </div>
                    <div className="emi-slider-row">
                      <label>Tenure: <strong>{emiTenure} years</strong></label>
                      <input type="range" min="5" max="30" step="1" value={emiTenure}
                        onChange={(e) => { setEmiTenure(+e.target.value); recalcEMI(emiRate, +e.target.value, emiDownPayment); }} />
                    </div>
                  </div>
                  <div className="emi-results">
                    <div className="emi-stat">
                      <div className="emi-stat-label">Monthly EMI</div>
                      <div className="emi-stat-value gradient-text">{formatINR(emiData.emi)}</div>
                    </div>
                    <div className="emi-stat">
                      <div className="emi-stat-label">Total Interest</div>
                      <div className="emi-stat-value">{formatINR(emiData.total_interest)}</div>
                    </div>
                    <div className="emi-stat">
                      <div className="emi-stat-label">Total Payment</div>
                      <div className="emi-stat-value">{formatINR(emiData.total_payment)}</div>
                    </div>
                    <div className="emi-stat">
                      <div className="emi-stat-label">Loan Amount</div>
                      <div className="emi-stat-value">{formatINR(emiData.principal)}</div>
                    </div>
                  </div>
                </div>
              )}

              {/* ── NEIGHBORHOOD STATS ── */}
              {neighborhoodStats && (
                <div className="glass-card">
                  <div className="model-header">
                    <span className="model-icon">📍</span>
                    <span className="model-title">{neighborhoodStats.location.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())} Market Snapshot</span>
                  </div>
                  <div className="nbhd-grid">
                    <div className="nbhd-stat"><div className="nbhd-label">Listings in dataset</div><div className="nbhd-value">{neighborhoodStats.total_listings}</div></div>
                    <div className="nbhd-stat"><div className="nbhd-label">Median price</div><div className="nbhd-value">{formatINR(neighborhoodStats.median_price)}</div></div>
                    <div className="nbhd-stat"><div className="nbhd-label">Avg ₹/sqft</div><div className="nbhd-value">₹{neighborhoodStats.avg_price_per_sqft.toLocaleString()}</div></div>
                    <div className="nbhd-stat"><div className="nbhd-label">Price range</div><div className="nbhd-value">{formatINR(neighborhoodStats.min_price)} – {formatINR(neighborhoodStats.max_price)}</div></div>
                  </div>
                </div>
              )}

              {/* ── SIMILAR PROPERTIES ── */}
              {similarProps.length > 0 && (
                <div className="glass-card">
                  <div className="model-header">
                    <span className="model-icon">🏘️</span>
                    <span className="model-title">Similar Properties from Dataset</span>
                  </div>
                  <div className="similar-list">
                    {similarProps.map((p, i) => (
                      <div key={i} className="similar-row">
                        <div>
                          <div className="similar-name">{p.location.replace(/_/g, " ")}</div>
                          <div className="similar-meta">{p.bhk} BHK · {Math.round(p.area_sqft)} sqft · ₹{Math.round(p.price_per_sqft).toLocaleString()}/sqft</div>
                        </div>
                        <div className="similar-price">{formatINR(p.price)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* ── TOP NEIGHBORHOODS ── */}
              {topNeighborhoods.length > 0 && (
                <div className="glass-card">
                  <div className="model-header">
                    <span className="model-icon">🏆</span>
                    <span className="model-title">Top Mumbai Neighborhoods (by ₹/sqft)</span>
                  </div>
                  <div className="top-list">
                    {topNeighborhoods.map((n, i) => (
                      <div key={i} className="top-row">
                        <div className="top-rank">#{i + 1}</div>
                        <div className="top-name">{n.location.replace(/_/g, " ")}</div>
                        <div className="top-psf">₹{Math.round(n.avg_psf).toLocaleString()}/sqft</div>
                        <div className="top-count">{n.count} listings</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* ── COMPARE NEIGHBORHOODS ── */}
              <div className="glass-card">
                <div className="model-header">
                  <span className="model-icon">⚖️</span>
                  <span className="model-title">Compare Neighborhoods</span>
                </div>
                {!showCompare ? (
                  <button className="feedback-cta" onClick={() => setShowCompare(true)}>
                    Compare {form.location.replace(/_/g, " ")} with other Mumbai areas →
                  </button>
                ) : (
                  <div>
                    <div className="compare-input-row">
                      <select className="input" onChange={(e) => {
                        if (e.target.value && !compareLocations.includes(e.target.value)) {
                          setCompareLocations([...compareLocations, e.target.value]);
                        }
                        e.target.value = "";
                      }}>
                        <option value="">+ Add neighborhood to compare...</option>
                        {locations.filter(l => !compareLocations.includes(l)).map(l => (
                          <option key={l} value={l}>{l.replace(/_/g, " ")}</option>
                        ))}
                      </select>
                    </div>
                    {compareLocations.length > 0 && (
                      <div className="compare-tags">
                        {compareLocations.map(l => (
                          <span key={l} className="compare-tag">
                            {l.replace(/_/g, " ")}
                            <button onClick={() => setCompareLocations(compareLocations.filter(x => x !== l))}>×</button>
                          </span>
                        ))}
                      </div>
                    )}
                    {compareLocations.length >= 2 && (
                      <button className="analyze-btn" style={{ marginTop: 12 }} onClick={runCompare}>
                        Compare {compareLocations.length} areas →
                      </button>
                    )}
                    {compareData.length > 0 && (
                      <div className="compare-grid">
                        {compareData.map((c, i) => (
                          <div key={i} className="compare-col">
                            <div className="compare-loc">{c.location.replace(/_/g, " ")}</div>
                            <div className="compare-stat"><span>Median:</span> <strong>{formatINR(c.median_price)}</strong></div>
                            <div className="compare-stat"><span>Avg/sqft:</span> <strong>₹{c.avg_price_per_sqft.toLocaleString()}</strong></div>
                            <div className="compare-stat"><span>Listings:</span> <strong>{c.total_listings}</strong></div>
                            <div className="compare-stat"><span>Avg area:</span> <strong>{c.avg_area} sqft</strong></div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* ACTIONS */}
              <div className="action-bar">
                <button className="action-btn" onClick={() => window.print()}>📄 Print Report</button>
                <button className="action-btn" onClick={() => {
                  navigator.clipboard.writeText(`PropWise Analysis: ${form.location.replace(/_/g, " ")} ${form.bhk}BHK ${form.area_sqft}sqft → ${formatINR(result.price_prediction.predicted_price)} (${result.price_prediction.model_accuracy}% accuracy)`);
                  alert("Copied to clipboard!");
                }}>📋 Copy Summary</button>
                <button className="action-btn" onClick={() => {
                  const text = encodeURIComponent(`Check out this AI property analysis: ${form.location.replace(/_/g, " ")} ${form.bhk}BHK predicted at ${formatINR(result.price_prediction.predicted_price)} via PropWise`);
                  window.open(`https://wa.me/?text=${text}`, "_blank");
                }}>💬 WhatsApp</button>
                <button className="action-btn primary" onClick={() => { setResult(null); window.scrollTo({ top: 0, behavior: "smooth" }); }}>↻ New Analysis</button>
              </div>
            </section>
          )}

          {/* FOOTER */}
          <footer className="footer">
            <div className="footer-text">
              PropWise v0.2 · 3 ML Models · Trained on real Mumbai property data · {serviceStatus?.stats?.total_predictions || 0} predictions made
            </div>
          </footer>
        </div>
      </div>
    </>
  );
}

// ─── Helper Components ─────────────────────────────────────────────────────

function Field({ label, hint, children }) {
  return (
    <div className="field">
      <label className="field-label">
        {label}
        {hint && <span className="field-hint">{hint}</span>}
      </label>
      {children}
    </div>
  );
}

function Checkbox({ label, checked, onChange }) {
  return (
    <label className={`checkbox ${checked ? "checked" : ""}`} onClick={() => onChange(checked ? 0 : 1)}>
      <span className="checkbox-box">{checked ? "✓" : ""}</span>
      <span>{label}</span>
    </label>
  );
}

function ModelBadge({ label, value, loaded }) {
  return (
    <div className={`model-badge ${loaded ? "" : "disabled"}`}>
      <div className="badge-label">{label}</div>
      <div className="badge-value">{value}</div>
    </div>
  );
}

function Tag({ color, children }) {
  return (
    <span className="tag" style={{ background: `${color}22`, borderColor: `${color}44`, color }}>
      {children}
    </span>
  );
}

function LoadingStep({ label, active, done }) {
  return (
    <div className={`load-step ${active ? "active" : ""} ${done ? "done" : ""}`}>
      <div className="load-dot">{done ? "✓" : ""}</div>
      <span>{label}</span>
    </div>
  );
}

// ─── Global CSS ────────────────────────────────────────────────────────────

const globalCSS = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Inter', system-ui, sans-serif;
    background: #050510;
    color: #e2e8f0;
    -webkit-font-smoothing: antialiased;
    line-height: 1.5;
  }

  .bg-gradient {
    min-height: 100vh;
    background: radial-gradient(ellipse at top, #0f172a 0%, #050510 60%);
    position: relative;
    overflow: hidden;
  }

  .bg-orbs { position: fixed; inset: 0; pointer-events: none; z-index: 0; }
  .orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.4;
    animation: float 20s ease-in-out infinite;
  }
  .orb-1 { width: 400px; height: 400px; background: #3b82f6; top: -100px; left: -100px; }
  .orb-2 { width: 500px; height: 500px; background: #8b5cf6; top: 30%; right: -200px; animation-delay: -7s; }
  .orb-3 { width: 350px; height: 350px; background: #06b6d4; bottom: -100px; left: 30%; animation-delay: -14s; }

  @keyframes float {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(30px, -50px) scale(1.1); }
    66% { transform: translate(-20px, 30px) scale(0.95); }
  }

  .container {
    position: relative;
    z-index: 1;
    max-width: 1080px;
    margin: 0 auto;
    padding: 32px 24px 80px;
  }

  /* HEADER */
  .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 60px; }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand-icon { font-size: 32px; }
  .brand-name { font-size: 22px; font-weight: 800; letter-spacing: -0.5px; }
  .brand-tagline { font-size: 11px; color: #64748b; letter-spacing: 1px; text-transform: uppercase; }

  .status-pill {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 14px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 100px; font-size: 12px; color: #cbd5e1;
  }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; }
  .status-dot.online { background: #10b981; box-shadow: 0 0 12px #10b981; animation: pulse 2s infinite; }
  .status-dot.offline { background: #ef4444; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

  /* HERO */
  .hero { text-align: center; margin-bottom: 50px; }
  .hero-title {
    font-size: clamp(32px, 6vw, 56px); font-weight: 900; letter-spacing: -2px;
    line-height: 1.05; margin-bottom: 16px; color: #f8fafc;
  }
  .gradient-text {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; background-size: 200% auto; animation: shimmer 4s linear infinite;
  }
  @keyframes shimmer { to { background-position: 200% center; } }
  .hero-subtitle { color: #94a3b8; font-size: 17px; margin-bottom: 28px; }

  .model-badges { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
  .model-badge {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 12px 18px; min-width: 100px;
    transition: all 0.3s ease;
  }
  .model-badge:hover { transform: translateY(-2px); border-color: #3b82f6; box-shadow: 0 8px 24px rgba(59,130,246,0.2); }
  .model-badge.disabled { opacity: 0.4; }
  .badge-label { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 4px; }
  .badge-value { font-size: 22px; font-weight: 800; color: #f1f5f9; font-family: 'JetBrains Mono', monospace; }

  /* GLASS CARD */
  .glass-card {
    background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 28px;
    margin-bottom: 20px;
    animation: slideUp 0.5s ease backwards;
  }
  @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

  .section-title { font-size: 14px; font-weight: 700; color: #cbd5e1; margin-bottom: 20px; letter-spacing: 0.5px; text-transform: uppercase; }

  /* FORM */
  .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 700px) { .form-grid { grid-template-columns: 1fr; } }

  .field { display: flex; flex-direction: column; gap: 8px; }
  .field-label { font-size: 13px; font-weight: 600; color: #cbd5e1; display: flex; align-items: center; gap: 8px; }
  .field-hint { font-size: 11px; color: #64748b; font-weight: 400; }

  .input {
    padding: 12px 14px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px; color: #f1f5f9; font-size: 14px; outline: none;
    font-family: inherit; transition: all 0.2s ease; width: 100%;
  }
  .input:hover { border-color: rgba(255,255,255,0.2); }
  .input:focus { border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59,130,246,0.15); }

  select.input { color-scheme: dark; }
  select.input option { background: #1e293b; color: #f1f5f9; }

  .number-input { position: relative; display: flex; align-items: center; }
  .number-input .prefix { position: absolute; left: 14px; color: #64748b; pointer-events: none; }
  .number-input .prefix ~ .input { padding-left: 28px; }
  .number-input .suffix { position: absolute; right: 14px; color: #64748b; font-size: 12px; }

  .pill-group { display: flex; gap: 6px; flex-wrap: wrap; }
  .pill {
    padding: 9px 14px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px; color: #94a3b8; font-size: 13px; font-weight: 600; cursor: pointer;
    transition: all 0.2s ease; font-family: inherit;
  }
  .pill:hover { border-color: rgba(255,255,255,0.2); color: #f1f5f9; }
  .pill.active { background: linear-gradient(135deg, #3b82f6, #2563eb); border-color: #3b82f6; color: #fff; box-shadow: 0 4px 12px rgba(59,130,246,0.3); }

  .checkbox-row { display: flex; gap: 10px; flex-wrap: wrap; }
  .checkbox {
    display: flex; align-items: center; gap: 8px; padding: 8px 12px;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px; cursor: pointer; transition: all 0.2s; font-size: 13px;
  }
  .checkbox:hover { border-color: rgba(255,255,255,0.2); }
  .checkbox.checked { background: rgba(59,130,246,0.1); border-color: #3b82f6; color: #fff; }
  .checkbox-box {
    width: 18px; height: 18px; border: 1.5px solid rgba(255,255,255,0.2); border-radius: 4px;
    display: flex; align-items: center; justify-content: center; font-size: 11px; color: #fff;
  }
  .checkbox.checked .checkbox-box { background: #3b82f6; border-color: #3b82f6; }

  /* AMENITIES */
  .amenities-section { margin-top: 24px; }
  .amenities-toggle {
    width: 100%; padding: 14px 18px; background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08); border-radius: 10px;
    display: flex; align-items: center; gap: 10px; color: #cbd5e1; font-size: 14px;
    font-weight: 600; cursor: pointer; transition: all 0.2s; font-family: inherit;
  }
  .amenities-toggle:hover { border-color: rgba(255,255,255,0.2); }
  .amenities-toggle .chevron { transition: transform 0.2s; }
  .amenities-toggle.open .chevron { transform: rotate(90deg); }
  .amenities-count { margin-left: auto; font-size: 12px; color: #64748b; font-weight: 500; }

  .amenities-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 8px; margin-top: 12px; animation: slideUp 0.3s ease; }
  .amenity-chip {
    display: flex; align-items: center; gap: 8px; padding: 10px 12px;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px; color: #94a3b8; font-size: 13px; cursor: pointer;
    transition: all 0.2s ease; font-family: inherit;
  }
  .amenity-chip:hover { border-color: rgba(255,255,255,0.2); color: #f1f5f9; }
  .amenity-chip.active { background: rgba(59,130,246,0.1); border-color: #3b82f6; color: #fff; }
  .amenity-icon { font-size: 16px; }
  .amenity-chip .check { margin-left: auto; color: #3b82f6; font-weight: 700; }

  /* ANALYZE BUTTON */
  .analyze-btn {
    width: 100%; margin-top: 28px; padding: 16px 24px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: #fff; border: none; border-radius: 12px;
    font-size: 15px; font-weight: 700; letter-spacing: 0.5px;
    cursor: pointer; transition: all 0.3s ease; font-family: inherit;
    box-shadow: 0 8px 24px rgba(59,130,246,0.3); position: relative; overflow: hidden;
  }
  .analyze-btn::before {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transform: translateX(-100%); transition: transform 0.6s;
  }
  .analyze-btn:hover:not(.disabled) { transform: translateY(-2px) scale(1.005); box-shadow: 0 12px 32px rgba(59,130,246,0.4); }
  .analyze-btn:hover:not(.disabled)::before { transform: translateX(100%); }
  .analyze-btn.disabled { opacity: 0.4; cursor: not-allowed; box-shadow: none; }

  .error-box { margin-top: 16px; padding: 12px 16px; background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); border-radius: 8px; color: #fca5a5; font-size: 13px; }

  /* LOADING */
  .loading-card { animation: slideUp 0.3s ease; }
  .loading-models { display: flex; flex-direction: column; gap: 14px; }
  .load-step {
    display: flex; align-items: center; gap: 14px; padding: 12px 16px;
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px; color: #64748b; font-size: 13px; font-weight: 500;
    transition: all 0.4s ease;
  }
  .load-step.active { background: rgba(59,130,246,0.08); border-color: #3b82f6; color: #f1f5f9; }
  .load-step.done { background: rgba(16,185,129,0.06); border-color: #10b981; color: #10b981; }
  .load-dot {
    width: 20px; height: 20px; border-radius: 50%; border: 2px solid currentColor;
    display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700;
  }
  .load-step.active .load-dot { animation: spin 1s linear infinite; border-style: dashed; }
  .load-step.done .load-dot { background: #10b981; border-color: #10b981; color: #fff; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* PRICE HERO */
  .price-hero {
    display: flex; align-items: center; justify-content: space-between; gap: 24px;
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(139,92,246,0.08));
    border-color: rgba(59,130,246,0.3);
  }
  .price-label { font-size: 11px; font-weight: 700; letter-spacing: 2px; color: #93c5fd; margin-bottom: 8px; }
  .price-value { font-size: clamp(38px, 7vw, 56px); font-weight: 900; line-height: 1; margin-bottom: 12px; letter-spacing: -2px; }
  .price-range { font-size: 14px; color: #cbd5e1; margin-bottom: 6px; }
  .price-sub { font-size: 12px; color: #64748b; font-family: 'JetBrains Mono', monospace; }
  .price-meter { flex-shrink: 0; }
  .meter-circle { position: relative; width: 120px; height: 120px; }
  .meter-svg { transform: rotate(-90deg); width: 100%; height: 100%; }
  .meter-bg { fill: none; stroke: rgba(255,255,255,0.05); stroke-width: 8; }
  .meter-fill { fill: none; stroke: url(#metergrad); stroke-width: 8; stroke-linecap: round; transition: stroke-dasharray 1.5s cubic-bezier(.4,0,.2,1); }
  .meter-text { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
  .meter-value { font-size: 24px; font-weight: 800; color: #fff; }
  .meter-label { font-size: 10px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }

  /* NEGOTIATION */
  .negotiation-card.warning { border-color: rgba(251,191,36,0.4); background: rgba(251,191,36,0.05); }
  .negotiation-card.alert { border-color: rgba(239,68,68,0.4); background: rgba(239,68,68,0.05); }
  .negotiation-card.success { border-color: rgba(16,185,129,0.4); background: rgba(16,185,129,0.05); }
  .neg-header { font-size: 13px; font-weight: 700; color: #cbd5e1; margin-bottom: 16px; }
  .neg-grid { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
  .neg-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
  .neg-value { font-size: 20px; font-weight: 800; color: #f1f5f9; }
  .neg-arrow { font-size: 24px; color: #64748b; }
  .neg-diff { margin-left: auto; padding: 8px 14px; border-radius: 8px; font-size: 18px; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
  .neg-diff.over { background: rgba(239,68,68,0.15); color: #fca5a5; }
  .neg-diff.under { background: rgba(16,185,129,0.15); color: #6ee7b7; }
  .neg-verdict { padding: 12px 14px; background: rgba(255,255,255,0.03); border-radius: 8px; font-size: 13px; color: #dbeafe; }

  /* ANALYSIS GRID */
  .analysis-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 700px) { .analysis-grid { grid-template-columns: 1fr; } }

  .model-card { animation: slideUp 0.5s ease backwards; }
  .model-header { display: flex; align-items: center; gap: 10px; margin-bottom: 18px; }
  .model-icon { font-size: 22px; }
  .model-title { font-size: 14px; font-weight: 700; color: #cbd5e1; letter-spacing: 0.5px; }

  .investment-score { display: flex; align-items: center; gap: 18px; margin-bottom: 20px; }
  .score-circle {
    width: 110px; height: 110px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; padding: 8px;
    transition: all 1s cubic-bezier(.4,0,.2,1);
  }
  .score-inner {
    width: 100%; height: 100%; border-radius: 50%; background: #050510;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
  }
  .score-num { font-size: 30px; font-weight: 900; color: #fff; line-height: 1; }
  .score-of { font-size: 11px; color: #64748b; }
  .verdict { font-size: 22px; font-weight: 800; margin-bottom: 4px; }
  .confidence { font-size: 12px; color: #94a3b8; }

  .prob-bars { display: flex; flex-direction: column; gap: 8px; }
  .prob-row { display: flex; align-items: center; gap: 10px; font-size: 12px; }
  .prob-label { width: 110px; color: #94a3b8; flex-shrink: 0; }
  .prob-bar { flex: 1; height: 8px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden; }
  .prob-fill { height: 100%; border-radius: 4px; transition: width 1.2s cubic-bezier(.4,0,.2,1); }
  .prob-pct { font-family: 'JetBrains Mono', monospace; color: #cbd5e1; min-width: 45px; text-align: right; font-size: 11px; }

  .anomaly-badge {
    display: inline-block; padding: 6px 12px; border-radius: 6px; font-size: 11px;
    font-weight: 800; letter-spacing: 1px; margin-bottom: 12px;
  }
  .model-card.normal .anomaly-badge { background: rgba(16,185,129,0.15); color: #6ee7b7; }
  .model-card.anomaly .anomaly-badge { background: rgba(239,68,68,0.15); color: #fca5a5; }
  .anomaly-score { font-size: 13px; color: #94a3b8; margin-bottom: 8px; font-family: 'JetBrains Mono', monospace; }
  .anomaly-score strong { color: #f1f5f9; }
  .anomaly-verdict { font-size: 13px; color: #cbd5e1; line-height: 1.6; }

  /* INSIGHTS */
  .insight-tags { display: flex; gap: 8px; flex-wrap: wrap; }
  .tag {
    padding: 6px 12px; border-radius: 100px; font-size: 12px; font-weight: 600;
    border: 1px solid; transition: all 0.2s;
  }

  /* FEEDBACK */
  .feedback-card { background: linear-gradient(135deg, rgba(139,92,246,0.05), rgba(59,130,246,0.05)); border-color: rgba(139,92,246,0.2); }
  .feedback-header { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
  .feedback-cta { width: 100%; padding: 12px 16px; background: rgba(139,92,246,0.1); border: 1px solid rgba(139,92,246,0.3); border-radius: 8px; color: #c4b5fd; font-size: 13px; cursor: pointer; transition: all 0.2s; font-family: inherit; }
  .feedback-cta:hover { background: rgba(139,92,246,0.2); }
  .feedback-form { display: flex; gap: 8px; }
  .feedback-form .input { flex: 1; }
  .feedback-submit { padding: 0 20px; background: #8b5cf6; color: #fff; border: none; border-radius: 8px; font-weight: 700; cursor: pointer; font-size: 13px; }
  .feedback-thanks { padding: 10px 14px; background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); border-radius: 8px; color: #6ee7b7; font-size: 13px; text-align: center; }

  /* SHAP EXPLAINABILITY */
  .explain-card { background: linear-gradient(135deg, rgba(168,85,247,0.05), rgba(59,130,246,0.05)); }
  .explain-intro { color: #cbd5e1; font-size: 13px; margin-bottom: 18px; }
  .explain-intro strong { color: #3b82f6; font-family: 'JetBrains Mono', monospace; }
  .explain-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 700px) { .explain-cols { grid-template-columns: 1fr; } }
  .explain-col { padding: 14px; background: rgba(0,0,0,0.2); border-radius: 8px; }
  .explain-col-title { font-size: 12px; font-weight: 700; letter-spacing: 1px; margin-bottom: 12px; text-transform: uppercase; }
  .explain-col-title.up { color: #6ee7b7; }
  .explain-col-title.down { color: #fca5a5; }
  .explain-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 13px; }
  .explain-row:last-child { border-bottom: none; }
  .explain-feat { color: #cbd5e1; }
  .explain-impact { font-family: 'JetBrains Mono', monospace; font-weight: 700; }
  .explain-impact.up { color: #6ee7b7; }
  .explain-impact.down { color: #fca5a5; }
  .explain-empty { color: #64748b; font-size: 12px; padding: 8px 0; }

  /* RENT VS BUY */
  .rentbuy-card { background: linear-gradient(135deg, rgba(16,185,129,0.05), rgba(251,191,36,0.05)); }
  .rentbuy-verdict { display: flex; align-items: center; gap: 12px; padding: 14px 18px; border: 1px solid; border-radius: 10px; margin-bottom: 18px; font-weight: 600; font-size: 14px; }
  .rentbuy-icon { font-size: 24px; }
  .rentbuy-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  @media (max-width: 700px) { .rentbuy-grid { grid-template-columns: 1fr; } }
  .rentbuy-col { padding: 16px; background: rgba(0,0,0,0.2); border-radius: 10px; border: 1px solid rgba(255,255,255,0.05); }
  .rentbuy-title { font-size: 13px; font-weight: 700; color: #f1f5f9; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.1); }
  .rentbuy-row { display: flex; justify-content: space-between; padding: 6px 0; font-size: 12px; }
  .rentbuy-row span { color: #94a3b8; }
  .rentbuy-row strong { color: #f1f5f9; font-family: 'JetBrains Mono', monospace; }
  .rentbuy-row.total { margin-top: 8px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 14px; }
  .rentbuy-row.total span { color: #cbd5e1; font-weight: 700; letter-spacing: 0.5px; }
  .rentbuy-row.total strong { color: #3b82f6; font-size: 16px; }
  .rentbuy-breakeven { margin-top: 14px; padding: 10px 14px; background: rgba(168,85,247,0.1); border: 1px solid rgba(168,85,247,0.3); border-radius: 8px; font-size: 13px; color: #c4b5fd; }
  .rentbuy-breakeven strong { color: #f1f5f9; }
  .rentbuy-hint { margin-top: 10px; font-size: 11px; color: #64748b; line-height: 1.5; }

  /* PRICE TREND */
  .trend-bars { display: flex; flex-direction: column; gap: 10px; }
  .trend-bar-row { display: grid; grid-template-columns: 60px 1fr auto; gap: 14px; align-items: center; }
  .trend-year { font-size: 13px; color: #94a3b8; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
  .trend-bar-track { flex: 1; height: 28px; background: rgba(0,0,0,0.2); border-radius: 6px; overflow: hidden; }
  .trend-bar-fill { height: 100%; background: linear-gradient(90deg, #3b82f6, #8b5cf6); border-radius: 6px; transition: width 1.5s cubic-bezier(.4,0,.2,1); }
  .trend-value { font-size: 13px; color: #f1f5f9; font-weight: 700; font-family: 'JetBrains Mono', monospace; min-width: 130px; text-align: right; }
  .trend-footer { margin-top: 14px; padding: 10px 14px; background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.3); border-radius: 8px; font-size: 13px; color: #93c5fd; text-align: center; }
  .trend-footer strong { color: #fff; }

  /* EMI CALCULATOR */
  .emi-card { background: linear-gradient(135deg, rgba(16,185,129,0.05), rgba(59,130,246,0.05)); }
  .emi-controls { display: flex; flex-direction: column; gap: 14px; margin-bottom: 20px; }
  .emi-slider-row label { display: block; font-size: 13px; color: #cbd5e1; margin-bottom: 8px; }
  .emi-slider-row label strong { color: #3b82f6; font-family: 'JetBrains Mono', monospace; }
  .emi-slider-row input[type="range"] { width: 100%; accent-color: #3b82f6; }
  .emi-results { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
  .emi-stat { padding: 14px; background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; }
  .emi-stat-label { font-size: 10px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
  .emi-stat-value { font-size: 18px; font-weight: 800; color: #f1f5f9; }

  /* NEIGHBORHOOD STATS */
  .nbhd-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
  .nbhd-stat { padding: 14px; background: rgba(0,0,0,0.2); border-radius: 8px; }
  .nbhd-label { font-size: 11px; color: #94a3b8; margin-bottom: 4px; }
  .nbhd-value { font-size: 18px; font-weight: 800; color: #f1f5f9; }

  /* SIMILAR PROPERTIES */
  .similar-list { display: flex; flex-direction: column; gap: 8px; }
  .similar-row { display: flex; justify-content: space-between; align-items: center; padding: 12px 14px; background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; transition: all 0.2s; }
  .similar-row:hover { border-color: #3b82f6; }
  .similar-name { font-weight: 700; color: #f1f5f9; text-transform: capitalize; }
  .similar-meta { font-size: 11px; color: #94a3b8; margin-top: 2px; font-family: 'JetBrains Mono', monospace; }
  .similar-price { font-size: 18px; font-weight: 800; color: #3b82f6; font-family: 'JetBrains Mono', monospace; }

  /* TOP NEIGHBORHOODS */
  .top-list { display: flex; flex-direction: column; gap: 6px; }
  .top-row { display: grid; grid-template-columns: 50px 1fr auto auto; gap: 14px; align-items: center; padding: 10px 14px; background: rgba(0,0,0,0.2); border-radius: 8px; }
  .top-rank { font-size: 16px; font-weight: 800; color: #fbbf24; font-family: 'JetBrains Mono', monospace; }
  .top-name { font-weight: 600; color: #f1f5f9; text-transform: capitalize; }
  .top-psf { font-size: 14px; font-weight: 700; color: #3b82f6; font-family: 'JetBrains Mono', monospace; }
  .top-count { font-size: 11px; color: #64748b; }

  /* COMPARE */
  .compare-input-row { margin-bottom: 12px; }
  .compare-tags { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
  .compare-tag { display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px; background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.3); border-radius: 100px; color: #93c5fd; font-size: 12px; text-transform: capitalize; }
  .compare-tag button { background: none; border: none; color: #93c5fd; cursor: pointer; font-size: 16px; padding: 0; line-height: 1; }
  .compare-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 16px; }
  .compare-col { padding: 14px; background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; }
  .compare-loc { font-size: 14px; font-weight: 700; color: #f1f5f9; text-transform: capitalize; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.1); }
  .compare-stat { display: flex; justify-content: space-between; padding: 4px 0; font-size: 12px; }
  .compare-stat span { color: #94a3b8; }
  .compare-stat strong { color: #f1f5f9; font-family: 'JetBrains Mono', monospace; }

  /* ACTIONS */
  .action-bar { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }
  .action-btn {
    padding: 10px 18px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px; color: #cbd5e1; font-size: 13px; font-weight: 600;
    cursor: pointer; transition: all 0.2s; font-family: inherit;
  }
  .action-btn:hover { border-color: rgba(255,255,255,0.2); transform: translateY(-1px); }
  .action-btn.primary { background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: #fff; border: none; }

  /* FOOTER */
  .footer { text-align: center; margin-top: 60px; padding-top: 30px; border-top: 1px solid rgba(255,255,255,0.05); }
  .footer-text { font-size: 12px; color: #475569; }

  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #050510; }
  ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #334155; }

  @media print { .bg-orbs, .action-bar, .feedback-card, .analyze-btn, .loading-card { display: none !important; } }

  /* ─── MOBILE RESPONSIVE ─── */
  @media (max-width: 480px) {
    .container { padding: 16px 12px 60px; }
    .header { flex-direction: column; gap: 12px; margin-bottom: 32px; text-align: center; }
    .hero { margin-bottom: 28px; }
    .hero-title { font-size: 28px; letter-spacing: -1px; }
    .hero-sub { font-size: 13px; }
    .model-badges { flex-wrap: wrap; justify-content: center; }
    .model-badge { font-size: 11px; padding: 6px 10px; }
    .glass-card { padding: 18px 14px; border-radius: 12px; }
    .field-label { font-size: 12px; }
    .field select, .field input { font-size: 14px; padding: 10px 12px; }
    .amenities-grid { grid-template-columns: 1fr 1fr; gap: 6px; }
    .analyze-btn { padding: 14px 16px; font-size: 14px; }
    .score-circle { width: 80px; height: 80px; }
    .score-num { font-size: 22px; }
    .investment-score { flex-direction: column; text-align: center; }
    .rentbuy-verdict { flex-direction: column; gap: 6px; }
    .action-bar { justify-content: center; }
    .action-btn { padding: 8px 12px; font-size: 11px; }
    .trend-bar-row { grid-template-columns: 40px 1fr auto; gap: 8px; }
    .explain-row { flex-wrap: wrap; gap: 4px; }
    .compare-grid { grid-template-columns: 1fr; }
    .similar-card { padding: 12px; }
    .cold-card { padding: 28px 18px; }
    .cold-card h2 { font-size: 18px; }
    .footer-text { font-size: 11px; }
  }

  @media (max-width: 360px) {
    .container { padding: 12px 8px 50px; }
    .hero-title { font-size: 24px; }
    .glass-card { padding: 14px 10px; }
    .model-badges { gap: 4px; }
    .amenities-grid { grid-template-columns: 1fr; }
  }

  /* Tablet tweaks */
  @media (min-width: 481px) and (max-width: 768px) {
    .container { padding: 24px 16px 60px; }
    .header { margin-bottom: 40px; }
    .glass-card { padding: 22px 18px; }
    .amenities-grid { grid-template-columns: repeat(3, 1fr); }
  }

  /* Fix iOS safari issues */
  @supports (-webkit-touch-callout: none) {
    .field select { font-size: 16px; /* prevents zoom on iOS */ }
    .field input { font-size: 16px; }
  }

  /* Touch-friendly targets */
  @media (pointer: coarse) {
    .checkbox { min-height: 44px; display: flex; align-items: center; }
    .action-btn { min-height: 44px; }
    .amenity-chip { min-height: 40px; }
  }

  /* COLD-START OVERLAY */
  .cold-overlay {
    position: fixed; inset: 0; z-index: 9999;
    background: rgba(2, 6, 23, 0.92); backdrop-filter: blur(12px);
    display: flex; align-items: center; justify-content: center; padding: 20px;
  }
  .cold-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
    border: 1px solid rgba(59,130,246,0.4); border-radius: 18px;
    padding: 40px 32px; max-width: 460px; width: 100%; text-align: center;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
  }
  .cold-card h2 { color: #fff; font-size: 22px; margin: 16px 0 10px; }
  .cold-card p { color: #94a3b8; font-size: 14px; line-height: 1.6; margin-bottom: 18px; }
  .cold-card .btn { margin-top: 8px; }
  .cold-spinner {
    width: 56px; height: 56px; margin: 0 auto;
    border: 4px solid rgba(59,130,246,0.2); border-top-color: #3b82f6;
    border-radius: 50%; animation: spin 0.9s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .cold-progress {
    width: 100%; height: 6px; background: rgba(255,255,255,0.05);
    border-radius: 999px; overflow: hidden; margin-top: 10px;
  }
  .cold-progress-bar {
    height: 100%; background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    transition: width 0.4s ease;
  }
  .cold-secs { font-size: 12px; color: #64748b; margin-top: 8px !important; }
`;
