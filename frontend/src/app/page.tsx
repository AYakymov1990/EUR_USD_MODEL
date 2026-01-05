type LogoCard = {
  label: string
  className: string
}

type TableRow = {
  company: string
  url: string
  createdBy: string
  address: string
  account: string
  badge: string
}

const logoCards: LogoCard[] = [
  {
    label: 'Stripe',
    className: 'right-[14%] top-7 rotate-[-4deg]',
  },
  {
    label: 'Github',
    className: 'right-6 top-20 rotate-[6deg]',
  },
  {
    label: 'Google',
    className: 'right-[20%] top-[38%] rotate-[-6deg]',
  },
  {
    label: 'Linkedin',
    className: 'left-16 bottom-[34%] rotate-[-2deg]',
  },
  {
    label: 'Figma',
    className: 'left-[32%] bottom-[20%] rotate-[8deg]',
  },
  {
    label: 'Anthropic',
    className: 'left-6 bottom-16 rotate-[4deg]',
  },
  {
    label: 'Airbnb',
    className: 'left-[46%] top-[50%] rotate-[-7deg]',
  },
  {
    label: 'Apple',
    className: 'left-[12%] top-[48%] rotate-[5deg]',
  },
]

const tableRows: TableRow[] = [
  {
    company: 'Qonto',
    url: 'qonto.com',
    createdBy: 'Jeff Williams',
    address: '18 Rue De Navarin',
    account: 'Phil Giller',
    badge: 'Q',
  },
  {
    company: 'Linkedin',
    url: 'linkedin.com',
    createdBy: 'Craig Federighi',
    address: '1226 Moises Caus',
    account: 'Phil Giller',
    badge: 'L',
  },
  {
    company: 'Slack',
    url: 'slack.com',
    createdBy: 'Eddy Cue',
    address: '1316 Dameon Moul',
    account: 'Phil Giller',
    badge: 'S',
  },
  {
    company: 'Notion',
    url: 'notion.com',
    createdBy: 'API - Key name',
    address: '1162 Sammy Creel',
    account: 'Phil Giller',
    badge: 'N',
  },
  {
    company: 'Github',
    url: 'github.com',
    createdBy: 'Workflow name',
    address: '110 Oswald June',
    account: 'Phil Giller',
    badge: 'G',
  },
  {
    company: 'Apple',
    url: 'apple.com',
    createdBy: 'Jeff Williams',
    address: '4517 Washington St',
    account: 'Tim Cook',
    badge: 'A',
  },
  {
    company: 'Figma',
    url: 'figma.com',
    createdBy: 'Tim Cook',
    address: '2118 Thomas Ave',
    account: 'Phil Giller',
    badge: 'F',
  },
  {
    company: 'Airbnb',
    url: 'airbnb.com',
    createdBy: 'Katherine Adams',
    address: '8574 Mission St',
    account: 'Phil Schiller',
    badge: 'A',
  },
  {
    company: 'Anthropic',
    url: 'anthropic.com',
    createdBy: 'Phil Schiller',
    address: '1905 Oswald Ave',
    account: 'Tim Cook',
    badge: 'AN',
  },
  {
    company: 'Notion Labs',
    url: 'notion.so',
    createdBy: 'Workflow name',
    address: '3691 Ranchview',
    account: 'Jeff Williams',
    badge: 'N',
  },
]

const columns = ['Companies', 'Url', 'Created By', 'Address', 'Account']

const avatarPalette = [
  'bg-sky-100 text-sky-700',
  'bg-amber-100 text-amber-700',
  'bg-emerald-100 text-emerald-700',
  'bg-indigo-100 text-indigo-700',
  'bg-rose-100 text-rose-700',
]

function LogoBadge({ card }: { card: LogoCard }) {
  const initials = card.label.slice(0, 2).toUpperCase()
  return (
    <div
      className={`pointer-events-none absolute flex items-center gap-3 rounded-2xl border border-neutral-200 bg-white px-4 py-2 shadow-[0_18px_45px_rgba(0,0,0,0.12)] ${card.className}`}
    >
      <div className="flex h-8 w-8 items-center justify-center overflow-hidden rounded-lg border border-neutral-200 bg-gradient-to-br from-neutral-50 to-neutral-200 text-[13px] font-semibold text-neutral-700">
        {initials}
      </div>
      <span className="text-base font-semibold text-neutral-900">
        {card.label}
      </span>
    </div>
  )
}

function Avatar({ name, colorIndex }: { name: string; colorIndex: number }) {
  const initials = name
    .split(' ')
    .map((part) => part[0])
    .join('')
    .slice(0, 2)
    .toUpperCase()

  const paletteClass = avatarPalette[colorIndex % avatarPalette.length]

  return (
    <span
      className={`flex h-9 w-9 items-center justify-center rounded-full border border-white text-sm font-semibold shadow-sm ${paletteClass}`}
    >
      {initials}
    </span>
  )
}

function TableRowItem({ row, index }: { row: TableRow; index: number }) {
  return (
    <div className="grid grid-cols-5 items-center gap-3 px-7 py-4 text-[15px] text-neutral-700">
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
        <Avatar name={row.createdBy} colorIndex={index} />
        <span className="text-[15px] font-semibold">{row.createdBy}</span>
      </div>
      <span className="text-[15px] text-neutral-700">{row.address}</span>
      <span className="text-[15px] text-neutral-700">{row.account}</span>
    </div>
  )
}

export default function Home() {
  return (
    <main className="relative isolate min-h-screen overflow-hidden bg-gradient-to-br from-[#f9fafc] via-[#f4f6fb] to-[#eef2f7] text-neutral-900">
      <div className="absolute inset-0 -z-20 bg-[radial-gradient(circle_at_1px_1px,#e4e7ee_1.2px,transparent_0)] [background-size:22px_22px]" />

      <div className="absolute right-[-30%] top-[-14%] -z-10 h-[155%] w-[78%] rotate-[16deg] rounded-[30px] border border-neutral-200/70 bg-white shadow-[0_48px_140px_rgba(0,0,0,0.18)]" />

      <div className="relative mx-auto flex max-w-7xl flex-col items-center gap-12 px-6 pb-24 pt-14 lg:flex-row lg:items-center lg:pt-24">
        <div className="relative w-full lg:w-[44%]">
          <div className="relative rotate-[-6deg]">
            <div
              className="absolute -left-6 bottom-5 h-10 w-[86%] rounded-[28px] bg-neutral-900/70 blur-[18px]"
              aria-hidden
            />
            <div className="rounded-[38px] border-[3px] border-neutral-900 bg-white px-10 py-12 shadow-[0_30px_95px_rgba(0,0,0,0.16)]">
              <p className="text-[32px] font-bold leading-tight text-neutral-700">
                The #1
              </p>
              <h1 className="mt-3 text-5xl font-black leading-[1.02] tracking-tight text-neutral-900 sm:text-[62px]">
                Open-Source CRM
              </h1>
              <p className="mt-6 max-w-xl text-xl leading-8 text-neutral-600">
                Modern, powerful, affordable platform to manage your customer
                relationships
              </p>
            </div>
          </div>
        </div>

        <div className="relative w-full lg:w-[56%]">
          <div className="relative flex justify-center">
            <div className="absolute left-[-90px] top-[16%] hidden rotate-[-12deg] flex-col gap-2 rounded-2xl border border-neutral-200 bg-white/95 px-3 py-4 shadow-[0_18px_50px_rgba(0,0,0,0.12)] lg:flex">
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
                {tableRows.map((row, index) => (
                  <TableRowItem key={`${row.company}-${index}`} row={row} index={index} />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {logoCards.map((card) => (
        <LogoBadge key={card.label} card={card} />
      ))}
    </main>
  )
}
