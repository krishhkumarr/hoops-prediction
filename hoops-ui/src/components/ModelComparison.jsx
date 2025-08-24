import React from 'react'
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from 'recharts'

export default function ModelComparison({ rows }){
  const data = rows?.map(r => ({
    Team: r.Team,
    rf: r.pred_rf,
    ridge: r.pred_ridge,
    baseline: r.pred_baseline,
  })) || []
  return (
    <div style={{height: 360}}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
          <XAxis dataKey="Team" tick={{ fill:'#9fb3c8' }} />
          <YAxis tick={{ fill:'#9fb3c8' }} domain={[0, 100]} />
          <Tooltip formatter={(v)=>[`${v}%`, 'Win %']} />
          <Legend />
          <Bar dataKey="rf" name="Random Forest" barSize={18} radius={[4,4,0,0]} fill="#5effa1" />
          <Bar dataKey="ridge" name="Ridge" barSize={18} radius={[4,4,0,0]} fill="#6aa0ff" />
          <Bar dataKey="baseline" name="Baseline" barSize={18} radius={[4,4,0,0]} fill="#f6c657" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
