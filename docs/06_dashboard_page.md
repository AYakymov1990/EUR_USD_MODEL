Command – Implement CRM Landing Page
Overview

Create the first (landing) page of the Trader CRM web app with a design inspired by Twenty.com’s hero section, but focus on functionality rather than exact pixel matching. The page should prominently display user account information, the latest trading signal, and two buttons labeled “LONG” and “SHORT” for confirming trades. According to the project docs, the frontend is already a Next.js 13 (App Router) application styled with Tailwind CSS, so build all new UI within that framework. In particular, edit src/app/page.tsx (and globals.css) to implement this page. The overall look/feel (fonts, colors, spacing) should roughly match the Twenty.com reference (an “Open-Source CRM” theme), but exact pixel-perfect alignment is not required; functionality and a clean responsive layout are higher priority.Use MCP Playwright to automate screenshots and comparisons!

Steps

Analyze the existing codebase and tech stack:

Confirm from the README that the frontend uses Next.js 13 (App Router) and Tailwind CSS. The landing page entrypoint is frontend/src/app/page.tsx and global styles are in src/app/globals.css.

Note that the docs explicitly say the new UI should mirror the Twenty.com hero section (a #1 Open-Source CRM theme). Use this as the design guide for layout and branding (for example, a header like “Trader CRM” or “Open-Source Trader CRM” to indicate the app’s purpose).

Set up the development environment:

Run the frontend locally:

cd frontend 
npm install
npm run dev


(This starts the Next.js app on http://localhost:3000.)

Also start the backend API (FastAPI) as described in the main README so that account and signal data endpoints are live. The frontend uses the NEXT_PUBLIC_API_BASE environment variable to know the API base URL (by default http://localhost:8000). Ensure this is set correctly (e.g. in a .env.local file) so the page can fetch real data.

Create the page layout and hero section:

In src/app/page.tsx, build the overall structure. Mimic the hero layout of Twenty.com: a bold headline, subheading, and call-to-action area. For example, use a large title like “Trader CRM” with a descriptive subtitle (e.g. “Manage trading signals and orders”), and a background or graphic if appropriate.

Below the hero, define separate sections or cards for Account Info and Signal Info. Use semantic HTML (<section>, <header>, etc.) and Tailwind utility classes to arrange these (for example, use flex or grid layouts). Keep the design clean – use Tailwind’s spacing and typography utilities to space elements similarly to the reference site.

Make sure to include some visible indication that this is a CRM/dashboard. For instance, include a small tagline or icon with the words “Trader CRM” or “Trading Dashboard” so users immediately recognize the app’s purpose.

Display account information:

Add UI elements (e.g. cards or panels) to show user account details. In React (client-side) code, fetch the account data from GET /account on the backend
GitHub
. You might do this in a useEffect hook or using Next.js data fetching.

Render key account fields on the page: for example, account balance, account ID, and whether it’s in demo/live mode. Format these clearly with labels (e.g. “Balance: $10,000”). Handle loading and error states (e.g. show “Loading…” or “Error loading account” if needed).

Display the trading signal:

Add a section for the latest trading signal. Use the API endpoint POST /signals/generate (to get a new prediction) or GET /signals/recent (to show the last signal)
GitHub
. Fetch a signal when the page loads (or on demand) and display the result.

Show at least the predicted action (e.g. Long or Short) and any relevant info (such as a confidence or reason text). For example: “Signal: Long (predicted price up)” or similar. Optionally include a timestamp or allow regenerating the signal with a button.

Implement LONG/SHORT order buttons:

Place two buttons labeled LONG and SHORT near the signal display. Style them distinctly (e.g. green/red) using Tailwind classes.

In the button click handlers, call POST /orders/market on the backend with { "action": "long" } or { "action": "short" } respectively
GitHub
. For example, use fetch or Axios to send the request.

While the request is in progress, disable the buttons or show a spinner. After the call returns, give feedback: e.g. display “Order sent” or show the updated account balance if available. Handle errors by showing an alert or message.

Style and responsiveness:

Use Tailwind CSS for all styling. Apply utility classes for layout (e.g. flex, grid, p-4, etc.), typography (e.g. text-2xl, font-bold), and colors to match the general palette of Twenty.com (for instance, use shades of blue and gray). You may edit src/app/globals.css for any global fonts or colors as needed
GitHub
.

Ensure the page is responsive: use responsive utility prefixes (like md:w-1/2, sm:text-lg, etc.) so it looks good on mobile and desktop. Check the layout at different window sizes and adjust spacing as needed.

Test and iterate:

Reload http://localhost:3000 and verify all sections render correctly. The account data should appear, the signal should load, and the LONG/SHORT buttons should work end-to-end.

Compare the layout to the design reference. The placement of headings, buttons, and text should be similar in a structural sense to Twenty.com’s hero (e.g. headline centered, content blocks in order), but some deviation is fine. Do not spend excessive time on perfect alignment; instead, ensure the page is clean, readable, and fully functional.

Fix any visual or functional issues: adjust margins, padding, or component sizes so the sections are well-proportioned. Repeat testing (including edge cases like API failures) until everything works smoothly.

Deliverables

Updated code in the frontend folder: the new landing page implemented in src/app/page.tsx (and any new components you create) with Tailwind styling.

Functional integration: The page should successfully fetch from /account and /signals, display that data, and send orders via /orders/market
GitHub
.

Design consistency: The layout and styling should follow the Twenty.com-inspired theme (CRM-focused) as described, using the existing Tailwind setup
GitHub
.

Verification: A brief confirmation (or screenshots) showing that on http://localhost:3000/, the account info, signal, and LONG/SHORT buttons all appear and function as expected. (These can be simple evidence that the page works and reflects the CRM context.)