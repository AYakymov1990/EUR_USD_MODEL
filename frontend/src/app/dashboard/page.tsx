'use client'

import { useEffect, useMemo, useState } from 'react'
import {
  getAccount,
  getConfig,
  getRecentSignals,
  generateSignal,
  submitOrder,
  type AccountResponse,
  type ConfigResponse,
  type SignalGenerateResponse,
  type SignalItem,
} from '@/lib/api'

type Status = 'idle' | 'loading' | 'error' | 'success'

type SignalRow = {
  company: string
  url: string
  createdBy: string
  address: string
  account: string
  badge: string
  action?: string
}

const FALLBACK_ROWS: SignalRow[] = [
  { company: 'Qonto', url: 'qonto.com', createdBy: 'Jeff Williams', address: '18 Rue De Navarin', account: 'Phil Giller', badge: 'Q' },
  { company: 'Linkedin', url: 'linkedin.com', createdBy: 'Craig Federighi', address: '1226 Moises Caus', account: 'Phil Giller', badge: 'L' },
  { company: 'Slack', url: 'slack.com', createdBy: 'Eddy Cue', address: '1316 Dameon Moul', account: 'Phil Giller', badge: 'S' },
  { company: 'Notion', url: 'notion.com', createdBy: 'API - Key name', address: '1162 Sammy Creel', account: 'Phil Giller', badge: 'N' },
  { company: 'Github', url: 'github.com', createdBy: 'Workflow name', address: '110 Oswald June', account: 'Phil Giller', badge: 'G' },
  { company: 'Airbnb', url: 'airbnb.com', createdBy: 'Katherine Adams', address: '8574 Mission St', account: 'Phil Schiller', badge: 'A' },
  { company: 'Figma', url: 'figma.com', createdBy: 'Tim Cook', address: '2118 Thomas Ave', account: 'Phil Giller', badge: 'F' },
]

const dottedBg = 'bg-[radial-gradient(circle_at_1px_1px,#e6e8ef_1px,transparent_0)] [background-size:22px_22px]'

function initials(name: string) {
  return name
    .split(' ')
    .map((p) => p[0])
    .join('')
    .slice(0, 2)
    .toUpperCase()
}

function useDashboardData() {
  const [config, setConfig] = useState<ConfigResponse | null>(null)
  const [account, setAccount] = useState<AccountResponse | null>(null)
  const [signals, setSignals] = useState<SignalItem[]>([])
  const [lastSignal, setLastSignal] = useState<SignalGenerateResponse | null>(null)
  const [status, setStatus] = useState<Status>('idle')
  const [orderStatus, setOrderStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | null>(null)

  const refresh = async () => {
    setStatus('loading')
    setError(null)
    try {
      const [cfg, acct, sigs] = await Promise.all([getConfig(), getAccount(), getRecentSignals(20)])
      setConfig(cfg)
      setAccount(acct)
      setSignals(sigs.items || [])
      setStatus('success')
    } catch (err) {
      setStatus('error')
      setError((err as Error).message)
    }
  }

  const generate = async () => {
    setStatus('loading')
    setError(null)
    try {
      const sig = await generateSignal()
      setLastSignal(sig)
      const sigs = await getRecentSignals(20)
      setSignals(sigs.items || [])
      setStatus('success')
    } catch (err) {
      setStatus('error')
      setError((err as Error).message)
    }
  }

  const sendOrder = async (direction: 'long' | 'short') => {
    setOrderStatus('loading')
    setError(null)
    try {
      await submitOrder(direction)
      setOrderStatus('success')
    } catch (err) {
      setOrderStatus('error')
      setError((err as Error).message)
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  return { config, account, signals, lastSignal, status, orderStatus, error, refresh, generate, sendOrder }
}

function TiltedTable({ rows }: { rows: SignalRow[] }) {
  const columns = ['Companies', 'Url', 'Created By', 'Address', 'Account']
  return (
    <div className="relative flex justify-center">
      <div className="absolute -left-16 top-16 hidden rotate-[-12deg] flex-col gap-2 rounded-2xl border border-neutral-200 bg-white/95 px-3 py-4 shadow-[0_18px_50px_rgba(0,0,0,0.12)] lg:flex">
        {['★', '☰', '⚙', '✚', '☑', '✦'].map((icon) => (
          <span
            key={icon}
            className="flex h-9 w-9 items-center justify-center rounded-xl bg-neutral-50 text-base font-semibold text-neutral-600 shadow-inner"
          >
            {icon}
          </span>
        ))}
      </div>

      <div className="relative rotate-[9deg] overflow-hidden rounded-[32px] border border-neutral-200 bg-white/95 shadow-[0_32px_100px_rgba(0,0,0,0.16)] backdrop-blur">
        <div className="grid grid-cols-5 gap-3 px-7 py-6 text-[11px] font-semibold uppercase tracking-[0.08em] text-neutral-500">
          {columns.map((column) => (
            <span key={column}>{column}</span>
          ))}
        </div>
        <div className="divide-y divide-neutral-200">
          {rows.map((row, index) => (
            <div
              key={`${row.company}-${index}`}
              className="grid grid-cols-5 items-center gap-3 px-7 py-4 text-[15px] text-neutral-700"
            >
              <div className="flex items-center gap-3">
                <span className="flex h-8 min-w-[32px] items-center justify-center rounded-lg bg-neutral-100 text-[11px] font-semibold uppercase text-neutral-700">
                  {row.badge}
                </span>
                <span className="text-[16px] font-semibold">{row.company}</span>
              </div>
              <span className="rounded-full border border-neutral-200 bg-neutral-50 px-3 py-1 text-xs font-semibold text-neutral-600">
                {row.url}
              </span>
              <div className="flex items-center gap-2 text-neutral-800">
                <span className="flex h-9 w-9 items-center justify-center rounded-full border border-white text-sm font-semibold shadow-sm bg-sky-100 text-sky-700">
                  {initials(row.createdBy)}
                </span>
                <span className="text-[15px] font-semibold">{row.createdBy}</span>
              </div>
              <span className="text-[15px] text-neutral-700">{row.address}</span>
              <span className="text-[15px] text-neutral-700">{row.account}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function BadgeCloud() {
  const cards = [
    { label: 'Stripe', className: 'right-[14%] top-7 rotate-[-4deg]' },
    { label: 'Github', className: 'right-6 top-20 rotate-[6deg]' },
    { label: 'Google', className: 'right-[18%] top-[38%] rotate-[-6deg]' },
    { label: 'Linkedin', className: 'left-16 bottom-[28%] rotate-[-2deg]' },
    { label: 'Figma', className: 'left-[32%] bottom-[16%] rotate-[8deg]' },
    { label: 'Anthropic', className: 'left-6 bottom-10 rotate-[4deg]' },
    { label: 'Airbnb', className: 'left-[46%] top-[54%] rotate-[-7deg]' },
    { label: 'Apple', className: 'left-[12%] top-[48%] rotate-[5deg]' },
  ]
  return (
    <>
      {cards.map((card) => (
        <div
          key={card.label}
          className={`pointer-events-none absolute flex items-center gap-2 rounded-2xl border border-neutral-200 bg-white px-3 py-2 text-sm font-semibold text-neutral-800 shadow-[0_18px_45px_rgba(0,0,0,0.12)] ${card.className}`}
        >
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-neutral-50 to-neutral-200 text-[11px] font-semibold text-neutral-700">
            {initials(card.label)}
          </span>
          {card.label}
        </div>
      ))}
    </>
  )
}

function HeroCard({ config }: { config: ConfigResponse | null }) {
  return (
    <div className="relative w-full lg:w-[46%]">
      <div className="relative rotate-[-6deg]">
        <div
          className="absolute -left-6 bottom-5 h-10 w-[86%] rounded-[28px] bg-neutral-900/50 blur-[18px]"
          aria-hidden
        />
        <div className="rounded-[38px] border-[3px] border-neutral-900 bg-white px-10 py-12 shadow-[0_30px_95px_rgba(0,0,0,0.16)]">
          <p className="text-[32px] font-bold leading-tight text-neutral-700">Trader CRM</p>
          <h1 className="mt-3 text-5xl font-black leading-[1.02] tracking-tight text-neutral-900 sm:text-[62px]">
            Open-Source CRM
          </h1>
          <p className="mt-6 max-w-xl text-xl leading-8 text-neutral-600">
            {config?.demo_mode ? 'DEMO режим на реплее данных.' : 'LIVE режим (practice) с OANDA.'}
          </p>
          <div className="mt-4 flex flex-wrap gap-3 text-sm text-neutral-700">
            <span className="rounded-full bg-neutral-100 px-3 py-1 font-semibold">
              {config?.instrument}/{config?.granularity}
            </span>
            <span className="rounded-full bg-neutral-100 px-3 py-1 font-semibold">
              {config?.demo_mode ? 'DEMO' : 'LIVE'}
            </span>
            <span className="rounded-full bg-neutral-100 px-3 py-1 font-semibold">API-driven</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function SignalCTA({
  onGenerate,
  onOrder,
  lastSignal,
  status,
}: {
  onGenerate: () => void
  onOrder: (d: 'long' | 'short') => void
  lastSignal: SignalGenerateResponse | null
  status: Status
}) {
  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-neutral-200 bg-white/80 px-5 py-4 shadow-sm backdrop-blur">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold uppercase text-neutral-500">Last signal</p>
          <p className="text-lg font-bold text-neutral-900">{lastSignal?.action ?? '—'}</p>
          <p className="text-sm text-neutral-600">{lastSignal?.ts ?? 'Нет сигналов'}</p>
        </div>
        <span className="rounded-full bg-neutral-100 px-3 py-1 text-xs font-semibold text-neutral-700">
          {status === 'loading' ? 'Loading…' : 'Ready'}
        </span>
      </div>
      <div className="flex flex-wrap gap-2">
        <button
          onClick={onGenerate}
          className="rounded-xl bg-neutral-900 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-neutral-800"
        >
          Получить сигнал
        </button>
        <button
          onClick={() => onOrder('long')}
          className="rounded-xl bg-emerald-600 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-emerald-500"
        >
          LONG
        </button>
        <button
          onClick={() => onOrder('short')}
          className="rounded-xl bg-rose-600 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-rose-500"
        >
          SHORT
        </button>
      </div>
      {lastSignal?.explanation && <p className="text-sm text-neutral-700">{lastSignal.explanation}</p>}
    </div>
  )
}

function AccountCard({ account, demo }: { account: AccountResponse | null; demo: boolean }) {
  const summary = useMemo(() => {
    if (!account) return []
    if ('account' in account && typeof account.account === 'object') {
      const acc = account.account as Record<string, any>
      return [
        { label: 'Alias', value: acc.alias },
        { label: 'Balance', value: acc.balance },
        { label: 'Currency', value: acc.currency },
        { label: 'Margin rate', value: acc.marginRate },
      ].filter((x) => x.value !== undefined)
    }
    return []
  }, [account])

  return (
    <div className="rounded-2xl border border-neutral-200 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Аккаунт</h3>
        <span className="rounded-full bg-neutral-100 px-3 py-1 text-xs font-semibold text-neutral-700">
          {demo ? 'DEMO' : 'LIVE'}
        </span>
      </div>
      {summary.length ? (
        <div className="mt-4 grid grid-cols-2 gap-3 text-sm text-neutral-800">
          {summary.map((item) => (
            <div key={item.label} className="rounded-xl bg-neutral-50 px-3 py-2">
              <p className="text-xs uppercase tracking-wide text-neutral-500">{item.label}</p>
              <p className="font-semibold text-neutral-900">{item.value as string}</p>
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-3 text-sm text-neutral-600">
          {demo ? 'Demo account summary' : 'Нет деталей аккаунта'}
        </p>
      )}
    </div>
  )
}

export default function DashboardPage() {
  const { config, account, signals, lastSignal, status, orderStatus, error, refresh, generate, sendOrder } = useDashboardData()

  const tableRows: SignalRow[] = useMemo(() => {
    if (!signals.length) return FALLBACK_ROWS
    return signals.slice(0, 8).map((s, idx) => ({
      company: s.payload?.company || `Signal ${idx + 1}`,
      url: s.payload?.url || 'n/a',
      createdBy: s.payload?.createdBy || s.action || 'Model',
      address: s.payload?.address || s.ts,
      account: s.payload?.account || '—',
      badge: initials(s.payload?.company || s.action || 'S'),
      action: s.action,
    }))
  }, [signals])

  return (
    <main className={`relative isolate min-h-screen overflow-hidden bg-gradient-to-br from-[#f9fafc] via-[#f4f6fb] to-[#eef2f7] text-neutral-900 ${dottedBg}`}>
      <div className="absolute right-[-30%] top-[-14%] -z-10 h-[155%] w-[78%] rotate-[16deg] rounded-[30px] border border-neutral-200/70 bg-white shadow-[0_48px_140px_rgba(0,0,0,0.18)]" />
      <div className="relative mx-auto flex max-w-7xl flex-col gap-12 px-6 pb-16 pt-14">
        <div className="flex flex-col items-start gap-8 lg:flex-row lg:items-start">
          <HeroCard config={config} />
          <div className="relative w-full lg:w-[54%]">
            <TiltedTable rows={tableRows} />
          </div>
          <BadgeCloud />
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          <SignalCTA onGenerate={generate} onOrder={sendOrder} lastSignal={lastSignal} status={status} />
          <AccountCard account={account} demo={!!config?.demo_mode} />
          <div className="rounded-2xl border border-neutral-200 bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Новости</h3>
              <span className="text-xs text-neutral-500">/news API</span>
            </div>
            <p className="mt-2 text-sm text-neutral-600">Подключите NewsAPI ключ, чтобы показать свежие EURUSD новости.</p>
            <button
              onClick={refresh}
              className="mt-4 rounded-xl border border-neutral-200 bg-neutral-50 px-3 py-2 text-sm font-semibold text-neutral-700 hover:border-neutral-300"
            >
              Refresh data
            </button>
          </div>
        </div>

        <div className="rounded-2xl border border-neutral-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Журнал сигналов</h3>
            <span className="text-xs text-neutral-500">последние {signals.length || FALLBACK_ROWS.length}</span>
          </div>
          <div className="mt-3 overflow-auto">
            <table className="min-w-full divide-y divide-neutral-200 text-sm">
              <thead className="bg-neutral-50">
                <tr>
                  <th className="px-3 py-2 text-left font-semibold text-neutral-600">ts</th>
                  <th className="px-3 py-2 text-left font-semibold text-neutral-600">action</th>
                  <th className="px-3 py-2 text-left font-semibold text-neutral-600">y_hat</th>
                  <th className="px-3 py-2 text-left font-semibold text-neutral-600">confidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-200">
                {(signals.length ? signals : []).map((s) => (
                  <tr key={`${s.ts}-${s.y_hat}`}>
                    <td className="whitespace-nowrap px-3 py-2 text-neutral-700">{s.ts}</td>
                    <td className="px-3 py-2">
                      <span className="rounded-full bg-neutral-100 px-2 py-1 text-xs font-semibold uppercase text-neutral-700">
                        {s.action}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-neutral-800">{s.y_hat?.toFixed?.(6)}</td>
                    <td className="px-3 py-2 text-neutral-800">{s.confidence?.toFixed?.(3)}</td>
                  </tr>
                ))}
                {signals.length === 0 && (
                  <tr>
                    <td className="px-3 py-2 text-neutral-600" colSpan={4}>
                      Журнал пуст. Сгенерируйте сигнал для заполнения.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {error && <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-rose-700">{error}</div>}

        <div className="flex items-center gap-3 text-xs text-neutral-500">
          <span>Статус: {status}</span>
          <span>Заявка: {orderStatus}</span>
          <span>API base: {process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'}</span>
        </div>
      </div>
    </main>
  )
}
