import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const METRIC_DISPLAY = [
  { key: 'cpu_utilization',          label: 'CPU %',       color: '185, 90%, 58%' },  // Cyan
  { key: 'memory_rss',               label: 'Mem RSS',      color: '270, 80%, 68%' },  // Purple
  { key: 'latency_p95',              label: 'p95 Lat (ms)', color: '40, 95%, 60%' },   // Orange
  { key: 'error_rate',               label: 'Err Rate',     color: '0, 82%, 60%' },    // Red
  { key: 'connection_pool_saturation', label: 'Pool Sat',   color: '210, 100%, 62%' }, // Blue
];

export default function MetricsChart({ history, services, selectedService, onSelect }) {
  const labels = history.map((_, i) => `${i + 1}`);

  const datasets = METRIC_DISPLAY.map((m, i) => {
    return {
      label: m.label,
      data: history.map(snap => snap[m.key] ?? 0),
      borderColor: `hsl(${m.color})`,
      backgroundColor: `hsla(${m.color}, 0.1)`,
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.3,
      fill: i === 0, // Only fill the first one to avoid mess
    };
  });

  const options = {
    animation: { duration: 200 },
    responsive: true,
    maintainAspectRatio: false,
    plugins: { 
      legend: { 
        labels: { color: 'hsl(220,10%,60%)', font: { size: 10 }, boxWidth: 12 },
        position: 'bottom'
      } 
    },
    scales: {
      x: { 
        ticks: { color: 'hsl(220,8%,40%)', font: { size: 9 } }, 
        grid: { color: 'hsla(220,20%,30%,0.3)' } 
      },
      y: { 
        ticks: { color: 'hsl(220,8%,40%)', font: { size: 9 } }, 
        grid: { color: 'hsla(220,20%,30%,0.3)' } 
      },
    },
  };

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ display: 'flex', justifyContent: 'flex-start', borderBottom: '1px solid var(--border)', background: 'rgba(0,0,0,0.1)' }}>
        {services.map(svc => (
          <button 
            key={svc}
            className={`btn ${svc === selectedService ? 'active' : ''}`}
            style={{ borderRadius: 0, border: 'none', borderRight: '1px solid var(--border)', borderBottom: svc === selectedService ? '2px solid var(--accent-blue)' : '2px solid transparent' }}
            onClick={() => onSelect(svc)}
          >
            {svc}
          </button>
        ))}
      </div>
      <div style={{ flex: 1, padding: '16px 24px', minHeight: 0 }}>
        {history.length > 0 ? (
           <Line data={{ labels, datasets }} options={options} />
        ) : (
           <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>
             Waiting for episode data...
           </div>
        )}
      </div>
    </div>
  );
}
