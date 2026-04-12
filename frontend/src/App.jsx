import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import ServiceHealthPanel from './components/ServiceHealthPanel';
import CausalDagViewer from './components/CausalDagViewer';
import MetricsChart from './components/MetricsChart';
import AIInsights from './components/AIInsights';
import ActionPanel from './components/ActionPanel';
import Leaderboard from './components/Leaderboard';
import './index.css';

const API = import.meta.env.VITE_API_URL || 'http://localhost:7860';

function App() {
  const [state, setState] = useState({
    taskId: 1,
    seed: 42,
    step: 0,
    done: false,
    services: [],
    metricHistory: {},
    lastObs: null,
    lastReward: null,
    actionLog: [],
    episodeScore: null,
    selectedService: null,
    mode: 'manual', // manual, agent, demo
    fullHistory: [],
    playbackStep: null,
  });

  const [status, setStatus] = useState({ text: 'Idle', mode: 'idle' });
  const [systemAlert, setSystemAlert] = useState(null);

  const fetchApi = async (endpoint, options = {}) => {
    try {
      const res = await fetch(`${API}${endpoint}`, options);
      if (!res.ok) throw new Error(`${endpoint} failed: ${res.status}`);
      return await res.json();
    } catch (e) {
      console.error(e);
      setStatus({ text: 'Error', mode: 'error' });
      // Use standard alert if no toast implemented
      alert(`API Error: ${e.message}`);
      throw e;
    }
  };

  const handleReset = async () => {
    setStatus({ text: 'Resetting…', mode: 'active' });
    try {
      const data = await fetchApi('/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: state.taskId, seed: state.seed }),
      });

      const obs = data.observation;
      const initialHistory = {};
      obs.services.forEach(svc => {
        initialHistory[svc] = obs.metrics?.[svc] ? [obs.metrics[svc]] : [];
      });

      setState(prev => ({
        ...prev,
        step: obs.step_number || 0,
        services: obs.services || [],
        lastObs: obs,
        done: false,
        actionLog: [],
        episodeScore: null,
        metricHistory: initialHistory,
        selectedService: obs.services?.[0] || null,
        fullHistory: [obs],
        playbackStep: prev.mode === 'demo' ? 0 : null,
      }));

      setStatus({ text: 'Running', mode: 'active' });
    } catch (e) {
      // Error handled in fetchApi
    }
  };

  const executeAgentStep = async () => {
    if (state.done || state.mode !== 'agent') return;
    setStatus({ text: 'AI Thinking…', mode: 'active' });
    try {
      const data = await fetchApi('/agent-step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ use_llm: true, model_name: "gpt-4o" }),
      });

      // Fetch step action reasoning if there and append to obs to easily track in history!
      if (data.action_taken?.reasoning) {
        data.observation.agent_reasoning = data.action_taken.reasoning;
      }
      processStepResponse(data.observation, data.reward, data.done, data.info, data.action_taken?.action_type, data.action_taken?.service_id);
    } catch (e) {
      // error handled
    }
  };

  const handleStep = async (actionType, serviceId, parameters) => {
    if (state.done) {
      alert('Episode is done. Reset to start again.');
      return;
    }

    setStatus({ text: 'Stepping…', mode: 'active' });
    try {
      const body = { action_type: actionType };
      if (serviceId) body.service_id = serviceId;
      if (parameters) body.parameters = parameters;

      const data = await fetchApi('/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      // Inject action details into observation directly so fullHistory has access to it for scrubber!
      data.observation.action_taken_type = actionType;
      data.observation.action_taken_service = serviceId;

      processStepResponse(data.observation, data.reward, data.done, data.info, actionType, serviceId);
    } catch (e) {
      // Error handled in fetchApi
    }
  };

  const processStepResponse = (obs, reward, done, info, actionType, serviceId) => {
    setState(prev => {
      const newHistory = { ...prev.metricHistory };
      obs.services.forEach(svc => {
        if (obs.metrics?.[svc]) {
          newHistory[svc] = [...(newHistory[svc] || []), obs.metrics[svc]];
        }
      });

      const newLog = [...prev.actionLog, {
        step: obs.step_number,
        action: actionType,
        service: serviceId || '—',
        reward: reward?.total ?? 0,
        breakdown: reward?.breakdown,
        reasoning: obs.agent_reasoning,
      }];

      const newScore = done && info?.episode_score != null ? info.episode_score : prev.episodeScore;

      if (done && newScore != null) {
        fetchApi('/leaderboard', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            run_id: Math.random().toString(36).substring(7),
            agent_name: prev.mode === 'agent' ? 'Hybrid AI' : 'Human Player',
            total_score: newScore,
            task_id: prev.taskId,
            steps_taken: obs.step_number,
            timestamp: new Date().toISOString()
          })
        }).catch(console.error);
      }

      return {
        ...prev,
        step: obs.step_number,
        lastObs: obs,
        lastReward: reward,
        done,
        metricHistory: newHistory,
        actionLog: newLog,
        episodeScore: newScore,
        fullHistory: [...prev.fullHistory, obs],
        playbackStep: prev.mode === 'demo' ? obs.step_number : null,
      };
    });

    setStatus({
      text: done ? 'Done' : `Step ${obs.step_number}`,
      mode: done ? 'idle' : 'active'
    });
  };

  // Agent Auto-pilot timer
  useEffect(() => {
    let timer;
    if (state.mode === 'agent' && !state.done && state.lastObs) {
      timer = setTimeout(executeAgentStep, 1500); // Step every 1.5s
    }
    return () => clearTimeout(timer);
  }, [state.mode, state.step, state.done, state.lastObs]);

  useEffect(() => {
    if (state.lastObs?.drift_alert) {
      setSystemAlert('⚠ Drift Alert: Predicted error rate exceeds safety threshold at t+15');
    } else {
      setSystemAlert(null);
    }
  }, [state.lastObs]);

  // Auto-start episode on first load
  useEffect(() => {
    handleReset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Decide what observation to show
  const currentViewObs = state.mode === 'demo' && state.playbackStep !== null
    ? state.fullHistory[state.playbackStep]
    : state.lastObs;

  return (
    <div className="app-container">
      {systemAlert && (
        <div className="alert-banner">
          <div className="pulse-dot"></div>
          <span>{systemAlert}</span>
          <button
            onClick={() => setSystemAlert(null)}
            style={{ marginLeft: 'auto', background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', fontSize: 16 }}
          >✕</button>
        </div>
      )}

      <Header
        state={state}
        setState={setState}
        status={status}
        onReset={handleReset}
      />

      <div className="main-grid">
        <ServiceHealthPanel
          state={{ ...state, lastObs: currentViewObs }}
          setState={setState}
        />

        <div className="panel-center">
          <div className="panel-header">
            <h2>🕸 Causal DAG</h2>
            <div style={{ display: 'flex', gap: 8 }}>
              <span className="chip chip-purple">PC Algorithm</span>
            </div>
          </div>
          <CausalDagViewer obs={currentViewObs} selectedService={state.selectedService} onSelectService={svc => setState(s => ({ ...s, selectedService: svc }))} />

          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 300, borderTop: '1px solid var(--border)' }}>
            <div className="panel-header" style={{ background: 'transparent' }}>
              <h2>📊 Metric History</h2>
            </div>
            <MetricsChart
              history={state.metricHistory[state.selectedService] || []}
              services={state.services}
              selectedService={state.selectedService}
              onSelect={svc => setState(s => ({ ...s, selectedService: svc }))}
            />
          </div>
        </div>

        <div className="panel-right">
          <div className="panel-header">
            <h2>🤖 AI Insights</h2>
            <span className="chip chip-blue" style={{ background: 'var(--bg-main)' }}>Step {currentViewObs?.step_number ?? state.step}</span>
          </div>

          <AIInsights obs={currentViewObs} />

          {state.mode !== 'demo' && (
            <ActionPanel
              services={state.services}
              alerts={currentViewObs?.alerts || []}
              onAction={handleStep}
              disabled={state.done || !currentViewObs || state.mode === 'agent'}
            />
          )}

          <Leaderboard />

        </div>
      </div>

      {state.mode === 'demo' && state.fullHistory.length > 0 && (
        <div className="demo-scrubber">
          <div style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>Demo Playback</div>
          <input
            type="range"
            className="scrubber-slider"
            min={0}
            max={state.fullHistory.length - 1}
            value={state.playbackStep ?? (state.fullHistory.length - 1)}
            onChange={e => setState(s => ({ ...s, playbackStep: parseInt(e.target.value, 10) }))}
          />
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>
            Step {state.playbackStep} / {state.fullHistory.length - 1}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
