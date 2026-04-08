import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

export default function CausalDagViewer({ obs, selectedService, onSelectService }) {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!obs || !obs.services) return;
    
    const dag = obs.causal_dag || {};
    const anomalyScores = obs.anomaly_scores || {};
    const services = obs.services || [];

    const W = svgRef.current.clientWidth || 600;
    const H = svgRef.current.clientHeight || 400;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g');

    // Arrow markers
    svg.append('defs').selectAll('marker')
      .data(['arrowhead', 'arrowhead-causal'])
      .enter()
      .append('marker')
      .attr('id', d => d)
      .attr('markerWidth', 8)
      .attr('markerHeight', 6)
      .attr('refX', 6)
      .attr('refY', 3)
      .attr('orient', 'auto')
      .append('polygon')
      .attr('points', '0 0, 8 3, 0 6')
      .attr('fill', d => d === 'arrowhead-causal' ? 'hsl(210,100%,62%)' : 'hsl(220,30%,50%)');

    const nodes = services.map(id => ({
      id,
      score: anomalyScores[id] ?? 0,
    }));

    const links = [];
    for (const [child, parents] of Object.entries(dag)) {
      for (const parent of (parents || [])) {
        if (services.includes(child) && services.includes(parent)) {
          links.push({ source: parent, target: child });
        }
      }
    }

    if (nodes.length === 0) return;

    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(120))
      .force('charge', d3.forceManyBody().strength(-280))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .force('collision', d3.forceCollide(48))
      .stop();

    for (let i = 0; i < 200; i++) sim.tick();

    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

    // Links
    g.selectAll('.dag-link')
      .data(links)
      .join('line')
      .attr('class', 'dag-link')
      .style('stroke', d => {
          if (d.source.score > 0.45 && d.target.score > 0.45) return 'var(--sev-critical)';
          return (d.source.score > 0.5 || d.target.score > 0.5) ? 'var(--accent-blue)' : 'var(--text-muted)';
      })
      .style('stroke-width', d => (d.source.score > 0.45 && d.target.score > 0.45) ? 4 : 2)
      .style('filter', d => (d.source.score > 0.45 && d.target.score > 0.45) ? 'drop-shadow(0 0 5px var(--sev-critical))' : 'none')
      .attr('x1', d => clamp(d.source.x, 36, W - 36))
      .attr('y1', d => clamp(d.source.y, 36, H - 36))
      .attr('x2', d => clamp(d.target.x, 36, W - 36))
      .attr('y2', d => clamp(d.target.y, 36, H - 36))
      .attr('marker-end', d => (d.source.score > 0.5 || d.target.score > 0.5) ? 'url(#arrowhead-causal)' : 'url(#arrowhead)');

    // Nodes
    const nodeG = g.selectAll('.dag-node')
      .data(nodes)
      .join('g')
      .attr('class', 'dag-node')
      .style('cursor', 'pointer')
      .attr('transform', d => `translate(${clamp(d.x, 36, W - 36)},${clamp(d.y, 36, H - 36)})`)
      .on('click', (_, d) => onSelectService(d.id));

    nodeG.append('circle')
      .attr('r', 30)
      .style('fill', 'var(--bg-elevated)')
      .style('stroke', d => d.id === selectedService ? 'var(--accent-blue)' : d.score > 0.65 ? 'var(--sev-critical)' : d.score > 0.4 ? 'var(--sev-warning)' : 'var(--border)')
      .style('stroke-width', d => d.id === selectedService ? 3 : 2);

    nodeG.append('text')
      .attr('y', 0)
      .attr('dy', '-0.3em')
      .attr('text-anchor', 'middle')
      .style('fill', 'var(--text-main)')
      .style('font-size', '10px')
      .style('pointer-events', 'none')
      .text(d => d.id.length > 8 ? d.id.slice(0, 7) + '…' : d.id);

    nodeG.append('text')
      .attr('y', 0)
      .attr('dy', '1.0em')
      .attr('text-anchor', 'middle')
      .style('fill', d => d.score > 0.65 ? 'var(--sev-critical)' : d.score > 0.4 ? 'var(--sev-warning)' : 'var(--text-muted)')
      .style('font-size', '10px')
      .style('font-weight', 'bold')
      .style('pointer-events', 'none')
      .text(d => `${(d.score * 100).toFixed(0)}%`);

  }, [obs, selectedService, onSelectService]);

  return (
    <div style={{ flex: 1, position: 'relative', display: 'flex' }}>
      <svg ref={svgRef} style={{ width: '100%', height: '100%' }}></svg>
      {(!obs || !obs.services || obs.services.length === 0) && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
          <div style={{ fontSize: 40, marginBottom: 16 }}>🕸</div>
          <div>Causal DAG will appear after reset</div>
          <div style={{ fontSize: 11 }}>PC algorithm runs at episode start</div>
        </div>
      )}
    </div>
  );
}
