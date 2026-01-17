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

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

async function fetchSignals(limit = 100): Promise<SignalItem[]> {
  const res = await fetch(`${API_BASE}/signals/recent?limit=${limit}`, { cache: 'no-store' })
  if (!res.ok) {
    throw new Error(`Signals request failed: ${res.status}`)
  }
  const data = await res.json()
  return data.items || []
}

const filters = [
  { value: 'all', label: 'Wszystkie' },
  { value: 'long', label: 'Long' },
  { value: 'short', label: 'Short' },
  { value: 'none', label: 'None' },
] as const

export default function SignalsPage() {
  const pathname = usePathname()
  const [signals, setSignals] = useState<SignalItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<(typeof filters)[number]['value']>('all')

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const items = await fetchSignals(200)
      setSignals(items)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  const filteredSignals = useMemo(() => {
    if (filter === 'all') return signals
    return signals.filter((s) => (s.action || '').toLowerCase() === filter)
  }, [signals, filter])

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
          <h1 className="mt-2 text-3xl font-black tracking-tight text-neutral-900 sm:text-4xl">Dziennik sygnałów</h1>
          <p className="mt-3 max-w-2xl text-lg text-neutral-600">
            Ostatnie sygnały modelu. Możesz filtrować po działaniu i odświeżać dane.
          </p>
          <div className="mt-4 flex flex-wrap items-center gap-3 text-sm text-neutral-700">
            {filters.map((f) => (
              <button
                key={f.value}
                onClick={() => setFilter(f.value)}
                className={`rounded-full border-2 px-4 py-2 font-semibold transition-all duration-150 ${
                  filter === f.value
                    ? 'border-neutral-900 bg-neutral-900 text-white shadow-[0_4px_0_#0f172a]'
                    : 'border-neutral-200 bg-white text-neutral-700 hover:border-neutral-300'
                }`}
              >
                {f.label}
              </button>
            ))}
            <button
              onClick={load}
              className="rounded-full border-2 border-neutral-900 bg-neutral-900 px-4 py-2 font-semibold text-white shadow-[0_6px_0_#0f172a] transition-all duration-150 hover:-translate-y-0.5 hover:shadow-[0_9px_0_#0f172a] active:translate-y-0 active:shadow-[0_3px_0_#0f172a] disabled:opacity-60"
              disabled={loading}
            >
              {loading ? 'Ładowanie…' : 'Odśwież'}
            </button>
          </div>
        </section>

        {error && (
          <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-rose-700 shadow-sm">{error}</div>
        )}

        <section className="overflow-hidden rounded-3xl border border-neutral-200 bg-white/85 shadow-[0_20px_60px_rgba(0,0,0,0.08)]">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-neutral-200 text-sm">
              <thead className="bg-neutral-50">
                <tr>
                  {['ts', 'action', 'y_hat', 'confidence', 'payload'].map((col) => (
                    <th key={col} className="px-4 py-3 text-left font-semibold uppercase tracking-wide text-neutral-600">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-100">
                {filteredSignals.map((s, idx) => (
                  <tr key={`${s.ts}-${idx}`} className="hover:bg-neutral-50/70 transition-colors">
                    <td className="whitespace-nowrap px-4 py-2 text-neutral-800">{s.ts}</td>
                    <td className="px-4 py-2">
                      <span
                        className={`rounded-full px-3 py-1 text-xs font-semibold uppercase ${
                          s.action === 'long'
                            ? 'bg-emerald-100 text-emerald-700'
                            : s.action === 'short'
                              ? 'bg-rose-100 text-rose-700'
                              : 'bg-neutral-100 text-neutral-700'
                        }`}
                      >
                        {s.action}
                      </span>
                    </td>
                    <td className="px-4 py-2 font-semibold text-neutral-900">{s.y_hat?.toFixed?.(6)}</td>
                    <td className="px-4 py-2 font-semibold text-neutral-900">{s.confidence?.toFixed?.(3)}</td>
                    <td className="px-4 py-2 text-neutral-700">
                      {s.payload && Object.keys(s.payload).length > 0 ? (
                        <pre className="max-h-28 overflow-auto rounded-lg bg-neutral-100 px-3 py-2 text-xs text-neutral-800">
                          {JSON.stringify(s.payload, null, 2)}
                        </pre>
                      ) : (
                        <span className="text-neutral-500">—</span>
                      )}
                    </td>
                  </tr>
                ))}
                {!loading && filteredSignals.length === 0 && (
                  <tr>
                    <td className="px-4 py-3 text-neutral-600" colSpan={5}>
                      Brak sygnałów dla wybranego filtra.
                    </td>
                  </tr>
                )}
                {loading && (
                  <tr>
                    <td className="px-4 py-3 text-neutral-600" colSpan={5}>
                      Ładowanie…
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </main>
  )
}
