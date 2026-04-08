import React from 'react';

export default function ServiceHealthPanel({ state, setState }) {
  const { services, lastObs, selectedService } = state;
  const metrics = lastObs?.metrics || {};
  const anomalyScores = lastObs?.anomaly_scores || {};
  const anomalyFlags = lastObs?.anomaly_flags || {};
  const dag = lastObs?.causal_dag || {};

  return (
    <div className="panel" id="panel-services" style={{ flexShrink: 0 }}>
      <div className="panel-header">
        <h2>🏥 Service Health</h2>
        <span className="chip chip-blue">{services.length} services</span>
      </div>
      <div className="panel-body" style={{ paddingBottom: 24, flex: 1, overflowY: 'auto' }}>
        {services.length === 0 ? (
          <div style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: '40px 0' }}>
            Press <strong>Reset Episode</strong> to start
          </div>
        ) : (
          services.map(svc => {
            const snap = metrics[svc] || {};
            const score = anomalyScores[svc] ?? 0;
            const flagged = anomalyFlags[svc];
            const parents = dag[svc] || [];

            const isAnomaly = score > 0.7;
            const isWarning = score > 0.45;
            const scoreClass = isAnomaly ? 'anomaly' : isWarning ? 'warning' : 'ok';
            const badgeClass = isAnomaly ? 'high' : isWarning ? 'medium' : 'low';

            const cpuClass = snap.cpu_utilization > 80 ? 'critical' : snap.cpu_utilization > 60 ? 'warning' : '';
            const memClass = snap.memory_rss > 1500 ? 'critical' : snap.memory_rss > 800 ? 'warning' : '';
            const errClass = snap.error_rate > 2 ? 'critical' : snap.error_rate > 0.8 ? 'warning' : '';
            
            const isRoot = parents.length === 0 && services.length > 1;

            return (
              <div 
                key={svc} 
                className={`service-card ${scoreClass} ${selectedService === svc ? 'active' : ''}`}
                onClick={() => setState(s => ({ ...s, selectedService: svc }))}
              >
                <div className="service-header">
                  <span className="service-name">
                    {svc}
                    {parents.length > 0 ? (
                      <span className="tag-dag-parent" title={`Parents: ${parents.join(', ')}`}>← {parents.join(', ')}</span>
                    ) : isRoot ? (
                      <span className="tag-dag-parent" style={{ background: 'hsla(0,70%,50%,0.15)', color: 'var(--sev-critical)' }}>root</span>
                    ) : null}
                  </span>
                  <span className={`anomaly-badge ${badgeClass}`}>{(score * 100).toFixed(0)}%</span>
                </div>
                <div className="metric-grid">
                  <div className="metric-item">
                    <span className="metric-label">CPU</span>
                    <span className={`metric-value ${cpuClass}`}>{snap.cpu_utilization?.toFixed(1) ?? '–'}%</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Mem RSS</span>
                    <span className={`metric-value ${memClass}`}>{snap.memory_rss?.toFixed(0) ?? '–'} MB</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">p95 Lat</span>
                    <span className={`metric-value ${errClass}`}>{snap.latency_p95?.toFixed(0) ?? '–'} ms</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Error Rate</span>
                    <span className={`metric-value ${errClass}`}>{snap.error_rate?.toFixed(2) ?? '–'}/s</span>
                  </div>
                </div>
                <div className="anomaly-bar">
                  <div 
                    className="anomaly-fill" 
                    style={{ 
                      width: `${(score * 100).toFixed(1)}%`, 
                      background: isAnomaly ? 'var(--sev-critical)' : isWarning ? 'var(--sev-warning)' : 'var(--sev-ok)' 
                    }}
                  ></div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
