import React from 'react';

export default function AIInsights({ obs }) {
  if (!obs) {
    return <div style={{ padding: 16, color: 'var(--text-muted)', fontSize: 12 }}>Waiting for episode…</div>;
  }

  const rcPred = obs.root_cause_prediction || '—';
  const probs = obs.root_cause_probabilities || {};
  const sortedProbs = Object.entries(probs).sort((a, b) => b[1] - a[1]);

  const shap = obs.shap_top5 || [];
  const maxShap = Math.max(...shap.map(([, v]) => Math.abs(v)), 0.001);

  const cf = obs.counterfactual_result;
  const forecast = obs.forecast;

  return (
    <>
      {/* Agent Reasoning */}
      {obs.agent_reasoning && (
        <>
          <div style={{ padding: 16, background: 'rgba(59, 130, 246, 0.1)', borderBottom: '1px solid var(--border)' }}>
            <div style={{ fontSize: 11, color: 'var(--accent-cyan)', fontWeight: 'bold', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>
               🤖 Agent Reasoning
            </div>
            <div style={{ fontSize: 13, lineHeight: 1.5 }}>
              {obs.agent_reasoning}
            </div>
          </div>
        </>
      )}

      {/* Root Cause Prediction */}
      <div style={{ padding: 16 }}>
        <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4 }}>Root Cause Prediction</div>
        <div style={{ fontSize: 18, fontWeight: 'bold', color: 'var(--accent-cyan)', marginBottom: 12 }}>{rcPred}</div>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {sortedProbs.map(([cls, p]) => (
            <div key={cls} style={{ display: 'flex', alignItems: 'center', fontSize: 11 }}>
              <span style={{ width: 100, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{cls}</span>
              <div style={{ flex: 1, height: 8, background: 'var(--bg-main)', borderRadius: 4, margin: '0 8px', overflow: 'hidden' }}>
                <div style={{ width: `${p * 100}%`, height: '100%', background: cls === rcPred ? 'var(--accent-cyan)' : 'var(--accent-blue)', transition: 'width 0.3s' }}></div>
              </div>
              <span style={{ width: 32, textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{(p * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>

      <hr style={{ border: 'none', borderTop: '1px solid var(--border)', margin: '0 16px' }} />

      {/* SHAP */}
      <div style={{ padding: 16 }}>
       <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
         <span style={{ fontSize: 12, fontWeight: 600 }}>⚡ SHAP Feature Attribution</span>
         <span className="chip chip-purple" style={{ fontSize: 9 }}>Top 5</span>
       </div>
       <div>
         {shap.length > 0 ? shap.map(([name, val]) => {
           const pct = (Math.abs(val) / maxShap * 100).toFixed(1);
           const shortName = name.replace(/__/g, ' ').slice(-28);
           return (
             <div key={name} className="shap-feature-row">
               <span className="shap-feat-name" title={name}>{shortName}</span>
               <div className="shap-bar-wrap">
                 <div className="shap-bar" style={{ width: `${pct}%`, background: val >= 0 ? 'var(--accent-purple)' : 'var(--accent-orange)' }}></div>
               </div>
               <span className="shap-val">{val.toFixed(4)}</span>
             </div>
           );
         }) : <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>Awaiting pipeline output…</div>}
       </div>
      </div>

      <hr style={{ border: 'none', borderTop: '1px solid var(--border)', margin: '0 16px' }} />

      {/* Counterfactuals */}
      <div style={{ padding: '16px 0' }}>
        <div style={{ margin: '0 16px 8px', fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.7, color: 'var(--text-muted)' }}>
          🔮 Counterfactual Simulation
        </div>
        <div className={`cf-panel ${cf ? (cf.harm_flag ? 'harm' : 'safe') : ''}`}>
          {cf ? (
            <>
              <div style={{ fontSize: 12, fontWeight: 600 }}>Action: {cf.action_type} on {cf.service_id}</div>
              <div style={{ fontSize: 24, fontWeight: 'bold', color: cf.predicted_resolution_probability >= 0.5 ? 'var(--sev-ok)' : 'var(--sev-critical)', margin: '4px 0' }}>
                {(cf.predicted_resolution_probability * 100).toFixed(1)}%
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Resolution probability</div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>CI: [{(cf.confidence_interval[0]*100).toFixed(0)}%, {(cf.confidence_interval[1]*100).toFixed(0)}%]</div>
              {cf.harm_flag && <div style={{ background: 'var(--sev-critical)', color: 'white', padding: '4px 8px', borderRadius: 4, marginTop: 8, fontSize: 11, fontWeight: 'bold' }}>⚠ HARM DETECTED — {cf.harm_description}</div>}
            </>
          ) : (
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>No query yet — use <code style={{ fontFamily: 'var(--font-mono)' }}>query_counterfactual</code> before acting</div>
          )}
        </div>
      </div>

      <hr style={{ border: 'none', borderTop: '1px solid var(--border)', margin: '0 16px' }} />

      {/* Forecast */}
      <div style={{ padding: 16 }}>
        <div style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.7, color: 'var(--text-muted)', marginBottom: 8 }}>
          📈 Temporal Forecast
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 11 }}>
          {forecast ? Object.keys(forecast.forecast_t5 || {}).slice(0, 4).map(svc => (
             <div key={svc} style={{ padding: 8, background: 'var(--bg-elevated)', borderRadius: 8, border: '1px solid var(--border)' }}>
               <div style={{ fontSize: 9, color: 'var(--text-muted)', marginBottom: 4, fontFamily: 'var(--font-mono)' }}>{svc}</div>
               <div style={{ fontSize: 12, fontWeight: 600, fontFamily: 'var(--font-mono)', color: (forecast.forecast_t5[svc]||0) > 2 ? 'var(--sev-critical)' : 'var(--accent-green)' }}>
                 t+5: {(forecast.forecast_t5[svc] || 0).toFixed(2)}
               </div>
               <div style={{ fontSize: 11, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                 t+15: {(forecast.forecast_t15[svc] || 0).toFixed(2)}
               </div>
             </div>
          )) : <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>Awaiting forecast data…</div>}
        </div>
      </div>
    </>
  );
}
