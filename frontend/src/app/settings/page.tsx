/* eslint-disable @next/next/no-img-element */
'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'
import { usePathname } from 'next/navigation'

const AUTO_KEY = 'autoSignalMode'

export default function SettingsPage() {
  const pathname = usePathname()
  const [autoMode, setAutoMode] = useState(false)

  useEffect(() => {
    if (typeof window === 'undefined') return
    const saved = localStorage.getItem(AUTO_KEY)
    setAutoMode(saved === '1')
  }, [])

  const toggleAuto = (next: boolean) => {
    setAutoMode(next)
    if (typeof window !== 'undefined') {
      localStorage.setItem(AUTO_KEY, next ? '1' : '0')
    }
  }

  return (
    <main className="relative isolate min-h-screen overflow-hidden bg-gradient-to-br from-[#f9fafc] via-[#f4f6fb] to-[#eef2f7] text-neutral-900">
      <div className="absolute inset-0 -z-20 bg-[radial-gradient(circle_at_1px_1px,#e4e7ee_1.2px,transparent_0)] [background-size:22px_22px]" />
      <div className="relative mx-auto flex max-w-6xl flex-col gap-8 px-6 pb-16 pt-10">
        <nav className="flex items-center gap-3">
          {[
            { href: '/', label: 'Dashboard' },
            { href: '/news', label: 'Новости' },
            { href: '/signals', label: 'Журнал' },
            { href: '/stats', label: 'Статистика' },
            { href: '/settings', label: 'Настройки' },
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
          <h1 className="mt-2 text-3xl font-black tracking-tight text-neutral-900 sm:text-4xl">Настройки</h1>
          <p className="mt-3 max-w-3xl text-lg text-neutral-600">
            Управление авто-получением сигналов. Режим сохраняется в браузере и подхватывается на dashboard.
          </p>
        </section>

        <section className="rounded-3xl border border-neutral-200 bg-white/90 p-6 shadow-[0_18px_55px_rgba(0,0,0,0.08)]">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-neutral-700">Авто-получение сигналов</p>
              <p className="text-sm text-neutral-600">
                При включении dashboard каждые 60 секунд запрашивает новый сигнал (если не идёт загрузка).
              </p>
            </div>
            <label className="flex items-center gap-3">
              <span className="text-sm font-semibold text-neutral-800">{autoMode ? 'Включено' : 'Выключено'}</span>
              <button
                onClick={() => toggleAuto(!autoMode)}
                className={`relative h-9 w-16 rounded-full border-2 transition-colors duration-150 ${
                  autoMode ? 'border-neutral-900 bg-neutral-900' : 'border-neutral-300 bg-neutral-200'
                }`}
                aria-pressed={autoMode}
              >
                <span
                  className={`absolute top-0.5 left-1 h-7 w-7 rounded-full bg-white shadow transition-transform duration-150 ${
                    autoMode ? 'translate-x-7' : 'translate-x-0'
                  }`}
                />
              </button>
            </label>
          </div>
        </section>
      </div>
    </main>
  )
}
