import React, { useState, useEffect } from 'react';

const API = import.meta.env.VITE_API_URL || 'http://localhost:7860';

export default function Leaderboard() {
  const [entries, setEntries] = useState([]);

  const fetchLB = async () => {
    try {
      const res = await fetch(`${API}/leaderboard`);
      if (res.ok) {
        const data = await res.json();
        setEntries(data.entries);
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchLB();
    // Poll every 5s just in case
    const intv = setInterval(fetchLB, 5000);
    return () => clearInterval(intv);
  }, []);

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', borderTop: '1px solid var(--border)' }}>
      <div className="panel-header" style={{ background: 'transparent' }}>
        <h2>🏆 Leaderboard</h2>
      </div>
      <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
        {entries.length === 0 ? (
          <div style={{ padding: 16, fontSize: 12, color: 'var(--text-muted)' }}>No recorded runs yet.</div>
        ) : (
          entries.map((entry, idx) => (
            <div key={idx} className="leaderboard-row">
              <span className="leaderboard-rank">#{idx + 1}</span>
              <span className="leaderboard-agent">{entry.agent_name} <span style={{fontSize: 10, color: 'var(--text-muted)'}}>Task {entry.task_id}</span></span>
              <span className="leaderboard-score">{entry.total_score.toFixed(3)}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
