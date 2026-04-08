import React, { useState, useEffect } from 'react';

const ACTIONS = [
  'query_counterfactual',
  'restart_service',
  'scale_service',
  'run_diagnostic',
  'silence_alert',
  'escalate_incident',
  'declare_resolution',
];

// Actions that require a service_id
const SERVICE_REQUIRED = ['restart_service', 'scale_service', 'run_diagnostic', 'query_counterfactual'];

// Actions that can be simulated via counterfactual
const SIMULABLE_ACTIONS = ['restart_service', 'scale_service', 'run_diagnostic'];

export default function ActionPanel({ services, alerts, onAction, disabled }) {
  const [actionType, setActionType] = useState('query_counterfactual');
  const [serviceId, setServiceId] = useState('');
  const [alertId, setAlertId] = useState('');
  const [simulatedAction, setSimulatedAction] = useState('restart_service');

  const needsService = SERVICE_REQUIRED.includes(actionType);
  const needsAlertId = actionType === 'silence_alert';
  const needsSimAction = actionType === 'query_counterfactual';

  const handleExecute = () => {
    let params = null;

    if (needsAlertId && alertId) {
      params = { alert_id: alertId };
    }

    if (needsSimAction) {
      params = { ...(params || {}), simulated_action: simulatedAction };
    }

    onAction(actionType, needsService ? (serviceId || null) : null, params);
  };

  // Gather available alert IDs from props
  const alertOptions = (alerts || [])
    .filter(a => !a.silenced)
    .map(a => a.alert_id);

  return (
    <div style={{ padding: 16 }}>
      <div style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.7, color: 'var(--text-muted)', marginBottom: 8 }}>
        ⚙ Dispatch Action
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <select 
          className="select-field" 
          value={actionType} 
          onChange={e => setActionType(e.target.value)}
          disabled={disabled}
          id="action-type-select"
        >
          {ACTIONS.map(a => (
            <option key={a} value={a}>
              {a === 'query_counterfactual' ? '🔮 ' : 
               a === 'restart_service' ? '🔄 ' : 
               a === 'scale_service' ? '📈 ' : 
               a === 'run_diagnostic' ? '🔍 ' : 
               a === 'silence_alert' ? '🔕 ' : 
               a === 'escalate_incident' ? '🚨 ' : '✅ '}{a}
            </option>
          ))}
        </select>

        {needsService && (
          <select 
            className="select-field" 
            value={serviceId} 
            onChange={e => setServiceId(e.target.value)}
            disabled={disabled}
            id="service-id-select"
          >
            <option value="">— service —</option>
            {services.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        )}

        {needsAlertId && (
          <select 
            className="select-field" 
            value={alertId} 
            onChange={e => setAlertId(e.target.value)}
            disabled={disabled}
            id="alert-id-select"
          >
            <option value="">— select alert —</option>
            {alertOptions.map(id => <option key={id} value={id}>{id}</option>)}
          </select>
        )}

        {needsSimAction && (
          <select 
            className="select-field" 
            value={simulatedAction} 
            onChange={e => setSimulatedAction(e.target.value)}
            disabled={disabled}
            id="simulated-action-select"
          >
            <option value="" disabled>— simulate action —</option>
            {SIMULABLE_ACTIONS.map(a => <option key={a} value={a}>{a}</option>)}
          </select>
        )}

        <button 
          className="btn btn-primary" 
          onClick={handleExecute}
          disabled={disabled || (needsService && !serviceId) || (needsAlertId && !alertId)}
          id="execute-action-btn"
        >
          ▶ Execute Action
        </button>
      </div>
    </div>
  );
}
