import React, { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import Leaderboard from './components/Leaderboard.jsx'
import ModelComparison from './components/ModelComparison.jsx'
import ConferenceChart from './components/ConferenceChart.jsx'
import Spline from '@splinetool/react-spline'

// Prefer build-time API URL. If not provided (e.g., Vercel rewrite), use relative base.
const API = import.meta.env.VITE_API_URL || ''

export default function App(){
  const [seasons, setSeasons] = useState([])
  const [season, setSeason] = useState(null)
  const [model, setModel] = useState('rf')
  const [lb, setLb] = useState([])
  const [comp, setComp] = useState([])
  const [conf, setConf] = useState([])

  useEffect(()=>{
    axios.get(`${API}/api/seasons`).then(res=>{
      setSeasons(res.data.seasons || [])
      const latest = (res.data.seasons || []).slice(-1)[0]
      setSeason(latest)
    }).catch(console.error)
  }, [])

  useEffect(()=>{
    if(!season) return
    axios.get(`${API}/api/leaderboard`, { params:{ season, model }})
      .then(res=> {
        setLb(res.data.leaderboard || [])
        if(res.data?.model && res.data.model !== model){
          setModel(res.data.model)
        }
      })
      .catch(console.error)
    axios.get(`${API}/api/model_comparison`, { params:{ season }})
      .then(res=> setComp(res.data.rows || [])).catch(console.error)
    axios.get(`${API}/api/conferences`, { params:{ season }})
      .then(res=> setConf(res.data.conferences || [])).catch(console.error)
  }, [season, model])

  return (
    <div className="container">
      <div style={{marginBottom:16, height:480, width:'100%'}}>
        <Spline scene="https://prod.spline.design/ND6apJxJqDZG0QGh/scene.splinecode" />
      </div>
      <div className="header">
        <div className="title">🏀 NCAA Championship Probabilities</div>
        <div className="controls">
          <select value={season || ''} onChange={e=>setSeason(Number(e.target.value))}>
            {seasons.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
          <select value={model} onChange={e=>setModel(e.target.value)}>
            <option value="rf">Random Forest</option>
            <option value="ridge">Ridge</option>
            <option value="baseline">Seed Baseline</option>
          </select>
        </div>
      </div>

      <div className="grid">
        <div className="card">
          <div className="card-header">
            <div className="card-title">Top Teams</div>
            <div className="chip">Model: {model}</div>
          </div>
          <div className="card-body">
            <Leaderboard rows={lb} />
          </div>
        </div>
        <div className="card">
          <div className="card-header">
            <div className="card-title">Model Comparison (Top 10)</div>
            <div className="chip">Season {season}</div>
          </div>
          <div className="card-body">
            <ModelComparison rows={comp} />
          </div>
        </div>
      </div>

      <div className="card" style={{marginTop:16}}>
        <div className="card-header">
          <div className="card-title">Conference Strength (avg. win prob)</div>
          <div className="chip">Season {season}</div>
        </div>
        <div className="card-body">
          <ConferenceChart rows={conf} />
          <div className="footer">Data from your CSV → served via <code>/api/*</code>. UI built with React + Recharts.</div>
        </div>
      </div>
    </div>
  )
}
