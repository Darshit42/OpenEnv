import React from 'react';

export default function Header({ state, setState, status, onReset }) {
  return (
    <header className="header">
      <div className="header-left">
        <div className="logo">🔬</div>
        <div className="header-title">
          <h1>OpenEnv SRE Intelligence</h1>
          <span>Causal Incident Diagnosis · Production RL Environment</span>
        </div>
      </div>

      <div className="header-right">
        <div className="mode-switch" style={{ marginRight: 16 }}>
          {['manual', 'agent', 'demo'].map(m => (
            <button 
              key={m}
              className={`mode-btn ${state.mode === m ? 'active' : ''}`}
              onClick={() => setState(prev => ({ ...prev, mode: m, playbackStep: m === 'demo' ? prev.step : null }))}
            >
              {m === 'manual' ? 'Manual' : m === 'agent' ? 'AI Agent' : 'Wow Demo'}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', background: 'var(--bg-elevated)', borderRadius: 6, overflow: 'hidden' }}>
          {[1, 2, 3].map(id => (
            <button 
              key={id}
              className={`btn ${state.taskId === id ? 'active' : ''}`}
              style={{ border: 'none', borderRadius: 0, padding: '4px 12px' }}
              onClick={() => setState(prev => ({ ...prev, taskId: id }))}
            >
              {id === 1 ? 'Mem Leak' : id === 2 ? 'Cascading' : 'Latency Chain'}
            </button>
          ))}
        </div>

        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-muted)' }}>
          Seed
          <input 
            type="number" 
            className="input-field" 
            value={state.seed} 
            min="0" max="9999" 
            style={{ width: 60 }}
            onChange={e => setState(s => ({ ...s, seed: parseInt(e.target.value, 10) || 42 }))} 
          />
        </label>

        <button className="btn btn-primary" onClick={onReset}>
          ⟳ Reset
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-muted)' }}>
          <div className={`status-dot ${status.mode}`}></div>
          <span>{status.text}</span>
        </div>
      </div>
    </header>
  );
}
