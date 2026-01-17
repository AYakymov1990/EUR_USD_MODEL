/* eslint-disable @next/next/no-img-element */
'use client'

import Link from 'next/link'
import { useEffect, useMemo, useState } from 'react'
import { usePathname } from 'next/navigation'

type SignalItem = {
  ts: string
  action: string
  y_hat: number
  confidence: number
  payload?: Record<string, any>
}

type OrderItem = {
  ts: string
  direction: string
  size: number
  status: string
  response?: Record<string, any>
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

async function fetchSignals(limit = 200): Promise<SignalItem[]> {
  const res = await fetch(`${API_BASE}/signals/recent?limit=${limit}`, { cache: 'no-store' })
  if (!res.ok) {
    throw new Error(`Signals request failed: ${res.status}`)
  }
  const data = await res.json()
  return data.items || []
}

async function fetchOrders(limit = 200): Promise<OrderItem[]> {
  const res = await fetch(`${API_BASE}/orders/recent?limit=${limit}`, { cache: 'no-store' })
  if (!res.ok) {
    throw new Error(`Orders request failed: ${res.status}`)
  }
  const data = await res.json()
  return data.items || []
}

function formatPct(v: number) {
  if (Number.isNaN(v)) return '0%'
  return `${(v * 100).toFixed(1)}%`
}

export default function StatsPage() {
  const pathname = usePathname()
  const [signals, setSignals] = useState<SignalItem[]>([])
  const [orders, setOrders] = useState<OrderItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [sigItems, ordItems] = await Promise.all([fetchSignals(200), fetchOrders(200)])
      setSignals(sigItems)
      setOrders(ordItems)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  const orderStats = useMemo(() => {
    const total = orders.length
    const longs = orders.filter((o) => o.direction?.toLowerCase() === 'long').length
    const shorts = orders.filter((o) => o.direction?.toLowerCase() === 'short').length
    const sent = orders.filter((o) => o.status === 'sent').length
    const avgSize =
      total > 0 ? orders.reduce((sum, o) => sum + (Number.isFinite(o.size) ? Number(o.size) : 0), 0) / total : 0
    const sentRate = total ? sent / total : 0
    const daily = new Map<string, number>()
    orders.forEach((o) => {
      const date = (o.ts || '').slice(0, 10)
      if (!date) return
      daily.set(date, (daily.get(date) || 0) + 1)
    })
    const dailyArr = Array.from(daily.entries()).sort((a, b) => a[0].localeCompare(b[0]))
    return { total, longs, shorts, sent, sentRate, avgSize, daily: dailyArr }
  }, [orders])

  const signalDaily = useMemo(() => {
    const daily = new Map<string, number>()
    signals.forEach((s) => {
      const date = (s.ts || '').slice(0, 10)
      if (!date) return
      daily.set(date, (daily.get(date) || 0) + 1)
    })
    return Array.from(daily.entries()).sort((a, b) => a[0].localeCompare(b[0]))
  }, [signals])

  const maxOrdersDaily = orderStats.daily.reduce((m, [, v]) => Math.max(m, v), 0) || 1
  const maxSignalDaily = signalDaily.reduce((m, [, v]) => Math.max(m, v), 0) || 1

  return (
    <main className="relative isolate min-h-screen overflow-hidden bg-gradient-to-br from-[#f9fafc] via-[#f4f6fb] to-[#eef2f7] text-neutral-900">
      <div className="absolute inset-0 -z-20 bg-[radial-gradient(circle_at_1px_1px,#e4e7ee_1.2px,transparent_0)] [background-size:22px_22px]" />
      <div className="relative mx-auto flex max-w-6xl flex-col gap-8 px-6 pb-16 pt-10">
        <nav className="flex items-center gap-3">
          {[
            { href: '/', label: 'Dashboard' },
            { href: '/news', label: 'Wiadomości' },
            { href: '/signals', label: 'Dziennik' },
            { href: '/stats', label: 'Statystyka' },
            { href: '/settings', label: 'Ustawienia' },
          ].map((item) => {
            const active = pathname === item.href
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`rounded-full border-2 px-4 py-2 text-sm font-semibold transition-all duration-150 ${
                  active
                    ? 'border-neutral-900 bg-neutral-900 text-white shadow-[0_4px_0_#0f172a]'
                    : 'border-neutral-200 bg-white text-neutral-700 hover:border-neutral-300'
                }`}
              >
                {item.label}
              </Link>
            )
          })}
        </nav>

        <section className="rounded-[28px] border-[2.5px] border-neutral-900 bg-white px-7 py-8 shadow-[0_22px_70px_rgba(0,0,0,0.12)]">
          <p className="text-sm font-semibold text-neutral-700">Trader CRM</p>
          <h1 className="mt-2 text-3xl font-black tracking-tight text-neutral-900 sm:text-4xl">Statystyka</h1>
          <p className="mt-3 max-w-3xl text-lg text-neutral-600">
            Metryki po sygnałach i zleceniach. P&L i win rate będą dokładne po integracji z realnymi transakcjami OANDA.
          </p>
          <div className="mt-4 flex flex-wrap items-center gap-3 text-sm text-neutral-700">
            <button
              onClick={load}
              className="rounded-full border-2 border-neutral-900 bg-neutral-900 px-4 py-2 font-semibold text-white shadow-[0_6px_0_#0f172a] transition-all duration-150 hover:-translate-y-0.5 hover:shadow-[0_9px_0_#0f172a] active:translate-y-0 active:shadow-[0_3px_0_#0f172a] disabled:opacity-60"
              disabled={loading}
            >
              {loading ? 'Ładowanie…' : 'Odśwież'}
            </button>
            <span className="rounded-full bg-neutral-100 px-3 py-1 text-xs font-semibold text-neutral-700">
              Źródło: {API_BASE.replace(/^https?:\/\//, '')}
            </span>
          </div>
        </section>

        {error && (
          <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-rose-700 shadow-sm">{error}</div>
        )}

        <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <StatCard title="Łącznie zleceń" value={orderStats.total.toString()} />
          <StatCard title="Long" value={orderStats.longs.toString()} accent="text-emerald-600" />
          <StatCard title="Short" value={orderStats.shorts.toString()} accent="text-rose-600" />
          <StatCard title="Sent rate" value={formatPct(orderStats.sentRate)} />
        </section>

        <section className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-3xl border border-neutral-200 bg-white/85 p-6 shadow-[0_20px_60px_rgba(0,0,0,0.08)]">
            <header className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-neutral-900">Zlecenia wg działania</h3>
              <span className="text-xs text-neutral-500">Ostatnie {orders.length}</span>
            </header>
            <div className="mt-4 grid grid-cols-3 gap-3 text-sm">
              {[
                { label: 'Long', value: orderStats.longs, color: 'bg-emerald-500' },
                { label: 'Short', value: orderStats.shorts, color: 'bg-rose-500' },
                { label: 'Sent', value: orderStats.sent, color: 'bg-neutral-800' },
              ].map((item) => (
                <div key={item.label} className="rounded-2xl border border-neutral-200 bg-neutral-50 p-4">
                  <p className="text-xs uppercase tracking-wide text-neutral-500">{item.label}</p>
                  <div className="mt-2 flex items-end gap-2">
                    <span className="text-xl font-bold text-neutral-900">{item.value}</span>
                    <span className={`h-2 w-8 rounded-full ${item.color}`} />
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-6 rounded-2xl border border-neutral-100 bg-neutral-50 p-4">
              <p className="text-xs uppercase tracking-wide text-neutral-500">Rozkład pasów</p>
              <div className="mt-3 flex gap-2">
                {[
                  { value: orderStats.longs, color: 'bg-emerald-500' },
                  { value: orderStats.shorts, color: 'bg-rose-500' },
                  { value: orderStats.sent, color: 'bg-neutral-800' },
                ]
                  .filter((b) => b.value > 0)
                  .map((b, idx) => (
                    <div
                      key={idx}
                      className={`h-3 rounded-full ${b.color}`}
                      style={{ width: `${Math.max(8, (b.value / Math.max(1, orderStats.total)) * 100)}%` }}
                    />
                  ))}
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-neutral-200 bg-white/85 p-6 shadow-[0_20px_60px_rgba(0,0,0,0.08)]">
            <header className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-neutral-900">Zlecenia wg dni</h3>
              <span className="text-xs text-neutral-500">Do {orderStats.daily.length} dni</span>
            </header>
            <div className="mt-4 space-y-3">
              {orderStats.daily.length === 0 && <p className="text-sm text-neutral-600">Brak danych po datach.</p>}
              {orderStats.daily.map(([date, count]) => (
                <div key={date} className="flex items-center gap-3">
                  <div className="w-20 text-sm font-semibold text-neutral-800">{date}</div>
                  <div className="h-3 flex-1 rounded-full bg-neutral-100">
                    <div
                      className="h-3 rounded-full bg-neutral-900"
                      style={{ width: `${Math.max(8, (count / maxOrdersDaily) * 100)}%` }}
                    />
                  </div>
                  <div className="w-10 text-right text-sm font-semibold text-neutral-800">{count}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="rounded-3xl border border-neutral-200 bg-white/85 p-6 shadow-[0_20px_60px_rgba(0,0,0,0.08)]">
          <header className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-neutral-900">Sygnały wg dni</h3>
            <span className="text-xs text-neutral-500">Do {signalDaily.length} dni</span>
          </header>
          <div className="mt-4 space-y-3">
            {signalDaily.length === 0 && <p className="text-sm text-neutral-600">Brak danych po datach.</p>}
            {signalDaily.map(([date, count]) => (
              <div key={date} className="flex items-center gap-3">
                <div className="w-20 text-sm font-semibold text-neutral-800">{date}</div>
                <div className="h-3 flex-1 rounded-full bg-neutral-100">
                  <div
                    className="h-3 rounded-full bg-neutral-700"
                    style={{ width: `${Math.max(8, (count / maxSignalDaily) * 100)}%` }}
                  />
                </div>
                <div className="w-10 text-right text-sm font-semibold text-neutral-800">{count}</div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </main>
  )
}

function StatCard({ title, value, accent }: { title: string; value: string; accent?: string }) {
  return (
    <div className="rounded-3xl border border-neutral-200 bg-white/90 p-5 shadow-[0_18px_55px_rgba(0,0,0,0.08)]">
      <p className="text-sm font-semibold text-neutral-600">{title}</p>
      <p className={`mt-2 text-2xl font-black text-neutral-900 ${accent ?? ''}`}>{value}</p>
    </div>
  )
}
