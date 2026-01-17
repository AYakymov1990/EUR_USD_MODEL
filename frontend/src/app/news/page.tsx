/* eslint-disable @next/next/no-img-element */
'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'
import { usePathname } from 'next/navigation'

type Article = {
  title?: string
  description?: string
  url?: string
  urlToImage?: string
  publishedAt?: string
  source?: { name?: string }
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

async function fetchNews(limit = 10): Promise<Article[]> {
  const res = await fetch(`${API_BASE}/news?limit=${limit}`, { cache: 'no-store' })
  if (!res.ok) {
    throw new Error(`News request failed: ${res.status}`)
  }
  const data = await res.json()
  return data.items || []
}

export default function NewsPage() {
  const pathname = usePathname()
  const [articles, setArticles] = useState<Article[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const items = await fetchNews(12)
      setArticles(items)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

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
          <h1 className="mt-2 text-3xl font-black tracking-tight text-neutral-900 sm:text-4xl">Wiadomości EUR/USD</h1>
          <p className="mt-3 max-w-2xl text-lg text-neutral-600">
            Strumień ostatnich wiadomości rynkowych. Źródło: NewsAPI przez backend ({API_BASE}).
          </p>
          <div className="mt-4 flex items-center gap-3 text-sm text-neutral-700">
            <span className="rounded-full bg-neutral-100 px-3 py-1 font-semibold">Aktualnie</span>
            <button
              onClick={load}
              className="rounded-full border-2 border-neutral-900 bg-neutral-900 px-4 py-2 text-sm font-semibold text-white shadow-[0_6px_0_#0f172a] transition-all duration-150 hover:-translate-y-0.5 hover:shadow-[0_9px_0_#0f172a] active:translate-y-0 active:shadow-[0_3px_0_#0f172a] disabled:opacity-60"
              disabled={loading}
            >
              {loading ? 'Ładowanie…' : 'Odśwież'}
            </button>
          </div>
        </section>

        {error && (
          <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-rose-700 shadow-sm">
            {error}
          </div>
        )}

        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {!loading && articles.length === 0 && (
            <div className="col-span-full rounded-2xl border border-neutral-200 bg-white/80 p-5 text-neutral-700 shadow-sm">
              Brak newsów. Sprawdź NEWS_API_KEY i dostęp do API.
            </div>
          )}
          {articles.map((art, idx) => (
            <article
              key={`${art.url}-${idx}`}
              className="group flex h-full flex-col justify-between gap-3 rounded-2xl border border-neutral-200 bg-white/85 p-4 shadow-[0_12px_36px_rgba(0,0,0,0.08)] transition-transform duration-150 hover:-translate-y-1"
            >
              {art.urlToImage && (
                <div className="overflow-hidden rounded-xl">
                  <img
                    src={art.urlToImage}
                    alt={art.title || ''}
                    className="h-32 w-full object-cover transition-transform duration-200 group-hover:scale-[1.02]"
                  />
                </div>
              )}
              <div className="flex-1 space-y-2">
                <p className="text-xs uppercase tracking-wide text-neutral-500">
                  {art.source?.name || 'Źródło'} · {art.publishedAt?.slice(0, 10) || ''}
                </p>
                <h2 className="text-lg font-semibold text-neutral-900">{art.title || 'Bez tytułu'}</h2>
                <p className="text-sm text-neutral-700 line-clamp-3">{art.description || 'Bez opisu'}</p>
              </div>
              {art.url && (
                <a
                  href={art.url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-2 text-sm font-semibold text-neutral-900 underline-offset-4 hover:underline"
                >
                  Otwórz źródło →
                </a>
              )}
            </article>
          ))}
        </section>
      </div>
    </main>
  )
}
