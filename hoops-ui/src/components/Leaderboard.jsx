import React from 'react'
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts'

export default function Leaderboard({ rows }){
  const top = rows?.slice(0, 20) || []
  return (
    <div>
      <div className="table-wrap">
        <table className="lb-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Team</th>
              <th>Conf</th>
              <th style={{textAlign:'right'}}>Win %</th>
            </tr>
          </thead>
          <tbody>
            {top.map((r, i)=> (
              <tr key={r.Team} style={r.champion ? {background:'#183247'} : undefined}>
                <td>{i+1}</td>
                <td>{r.Team}</td>
                <td className="muted">{r.conference}</td>
                <td style={{textAlign:'right',fontWeight:800}}>{Number(r.win_probability).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
