import React from 'react'
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts'

export default function ConferenceChart({ rows }){
  const data = rows?.slice(0, 14) || []
  return (
    <div style={{height: 420}}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 16, left: 16, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
          <XAxis type="number" tick={{ fill:'#9fb3c8' }} />
          <YAxis dataKey="conference" type="category" tick={{ fill:'#9fb3c8' }} width={120} />
          <Tooltip />
          <Bar dataKey="pred_rf" name="Random Forest" radius={8} />
          <Bar dataKey="pred_ridge" name="Ridge" radius={8} />
          <Bar dataKey="pred_baseline" name="Baseline" radius={8} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
