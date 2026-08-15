import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { TrendingDown, DollarSign, Target, Users } from "lucide-react";

const contractData = [
  { name: "Month-to-month", churn: 37.7, customers: 2748 },
  { name: "One year", churn: 23.2, customers: 1253 },
  { name: "Two year", churn: 20.9, customers: 999 },
];

const tenureData = [
  { name: "0-6 mo", churn: 41.6 },
  { name: "6-12 mo", churn: 27.1 },
  { name: "1-2 yr", churn: 29.3 },
  { name: "2-4 yr", churn: 27.6 },
  { name: "4+ yr", churn: 17.3 },
];

const driverData = [
  { name: "Monthly charges", value: 18.7 },
  { name: "Total charges", value: 18.2 },
  { name: "Tenure", value: 17.4 },
  { name: "2-year contract", value: 5.9 },
  { name: "Electronic check", value: 5.4 },
];

const INK = "#1c2a3a";
const RISK = "#c65d3b";
const SAFE = "#3b7a6b";
const MUTE = "#8a94a3";

function KpiCard({ icon: Icon, label, value, sub, accent }) {
  return (
    <div style={{
      background: "#fff", borderRadius: 10, padding: "18px 20px",
      border: "1px solid #e7e5df", flex: 1, minWidth: 150,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <Icon size={16} color={accent} strokeWidth={2.25} />
        <span style={{ fontSize: 12, color: MUTE, letterSpacing: "0.04em", textTransform: "uppercase", fontWeight: 600 }}>{label}</span>
      </div>
      <div style={{ fontSize: 26, fontWeight: 700, color: INK, fontFamily: "'IBM Plex Mono', monospace" }}>{value}</div>
      <div style={{ fontSize: 12.5, color: MUTE, marginTop: 4 }}>{sub}</div>
    </div>
  );
}

export default function ChurnDashboard() {
  const [tab, setTab] = useState("contract");
  const data = tab === "contract" ? contractData : tenureData;

  return (
    <div style={{
      fontFamily: "'Inter', -apple-system, sans-serif", background: "#f6f4ef",
      padding: 24, borderRadius: 12, maxWidth: 780, margin: "0 auto",
    }}>
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: RISK, letterSpacing: "0.08em", textTransform: "uppercase" }}>
          Telco Retention Analytics
        </div>
        <h2 style={{ margin: "4px 0 2px", fontSize: 22, color: INK, fontWeight: 700 }}>
          Customer Churn Risk Dashboard
        </h2>
        <div style={{ fontSize: 13, color: MUTE }}>5,000 customers · scored on churn probability</div>
      </div>

      <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
        <KpiCard icon={Users} label="Churn rate" value="30.7%" sub="1,535 of 5,000 customers" accent={RISK} />
        <KpiCard icon={DollarSign} label="Revenue at risk" value="$841K" sub="annualized, month-to-month segment" accent={RISK} />
        <KpiCard icon={Target} label="Top-risk precision" value="69.6%" sub="of top-20% flagged, actually churn" accent={SAFE} />
        <KpiCard icon={TrendingDown} label="Campaign ROI" value="10.9x" sub="$15K spend → $179K saved" accent={SAFE} />
      </div>

      <div style={{ background: "#fff", borderRadius: 10, border: "1px solid #e7e5df", padding: "18px 20px", marginBottom: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <span style={{ fontSize: 13.5, fontWeight: 600, color: INK }}>Churn rate by segment</span>
          <div style={{ display: "flex", gap: 6 }}>
            {["contract", "tenure"].map(t => (
              <button key={t} onClick={() => setTab(t)} style={{
                fontSize: 12, padding: "5px 12px", borderRadius: 6, border: "1px solid #e7e5df",
                background: tab === t ? INK : "#fff", color: tab === t ? "#fff" : MUTE,
                cursor: "pointer", fontWeight: 600, textTransform: "capitalize",
              }}>{t}</button>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={190}>
          <BarChart data={data} margin={{ left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eee" vertical={false} />
            <XAxis dataKey="name" tick={{ fontSize: 11.5, fill: MUTE }} axisLine={{ stroke: "#ddd" }} tickLine={false} />
            <YAxis tick={{ fontSize: 11, fill: MUTE }} unit="%" axisLine={false} tickLine={false} />
            <Tooltip formatter={(v) => [`${v}%`, "Churn rate"]} contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e7e5df" }} />
            <Bar dataKey="churn" radius={[5, 5, 0, 0]}>
              {data.map((d, i) => (
                <Cell key={i} fill={d.churn > 30 ? RISK : d.churn > 22 ? "#d9a05b" : SAFE} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ background: "#fff", borderRadius: 10, border: "1px solid #e7e5df", padding: "18px 20px" }}>
        <div style={{ fontSize: 13.5, fontWeight: 600, color: INK, marginBottom: 14 }}>Top churn drivers (model feature importance)</div>
        <ResponsiveContainer width="100%" height={170}>
          <BarChart data={driverData} layout="vertical" margin={{ left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eee" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 11, fill: MUTE }} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11.5, fill: INK }} width={110} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e7e5df" }} />
            <Bar dataKey="value" fill={INK} radius={[0, 5, 5, 0]} barSize={16} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ fontSize: 11.5, color: MUTE, marginTop: 14, lineHeight: 1.5 }}>
        Recommendation: prioritize retention outreach on month-to-month, sub-6-month-tenure customers
        paying by electronic check — this segment shows 44.7% churn vs. 30.7% baseline.
      </div>
    </div>
  );
}
