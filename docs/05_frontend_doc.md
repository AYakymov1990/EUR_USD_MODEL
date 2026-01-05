Command – Build New Frontend for Trader CRM
Overview

We need to create a standalone web frontend for our Trader CRM (EUR/USD model) that exactly matches the design of a given reference site (link provided separately). This will replace the current Streamlit UI with a more conventional web app (e.g. React/Next.js or similar). The goal is a pixel-perfect copy of the original design: all layouts, typography, colors, and components must match exactly. We will assemble a team (front-end developers, a UI/UX designer, a QA/tester, etc.) to execute this. The development will use MCP Playwright for automated screenshot comparisons, iteratively fixing discrepancies until the new frontend is visually identical to the original. The frontend code should live in a new directory (e.g. /frontend), separate from the Python backend, to avoid conflicts and aid navigation
nairihar.medium.com
netguru.com
. We will also follow modern best practices: use a component-based framework (like React with Tailwind CSS), maintain clean code (SOLID principles, meaningful names)
medium.com
, and enforce coding standards with tools like ESLint/Prettier
bacancytechnology.com
.

Steps

1.Assemble Implementation Team. Identify and assign roles needed for the frontend project (e.g., one or more front-end developers skilled in React/Next.js or Vue, a UI/UX designer for layouts, a QA/test engineer for visual testing, and a DevOps/CI specialist). Define responsibilities and communication channels. A cross-functional team ensures design, development, and testing are well-coordinated.

2.Select Frontend Technology. Choose a web framework that fits our needs. Popular options include React (often with Next.js for full-featured apps), Vue, or Angular. For this project, React (with TypeScript) is recommended due to its component-based architecture and strong ecosystem. We will use a utility-first CSS framework like Tailwind CSS or a CSS module approach to achieve precise styling. (React’s emphasis on reusable components and hooks aligns well with our project scope
medium.com
.) Enforce strict coding standards: enable TypeScript strict mode and use linting (ESLint) and formatting (Prettier) for consistency
bacancytechnology.com
.

3.Set Up Project Structure. Create a new top-level folder (e.g. frontend/ or ui/) in the repository for the web app
nairihar.medium.com
. Initialize the project (e.g. using create-react-app, Next.js starter, Vite, etc.) inside this folder. Organize it according to best practices: include directories like components/, pages/ or views/, hooks/, assets/, and services/
netguru.com
. This clear separation ensures backend and frontend code don’t interfere. Configure absolute imports or path aliases (via jsconfig.json or tsconfig.json) to simplify imports across components
netguru.com
. Commit the initial structure to version control.

4.Capture Reference Design. Open the provided reference website link in a browser. Use the MCP Playwright tool to take a full-page screenshot (e.g. mcp playwright or an equivalent script). Save this as original.png. This image will serve as our baseline. Inspect the page structure – identify all major sections (headers, nav bars, content blocks, footers, etc.). Document fonts, colors, and spacing if needed. This visual reference guides our implementation.

5.Build Initial Page Layout. In the new frontend project, create the first page (e.g. pages/index.tsx for Next.js or App.js for CRA). Reproduce the structure of the reference site one section at a time: create React components for each major block (header, hero/banner, features, charts, footer, etc.). Use placeholder content or copy text from the reference. At this stage, focus on the layout and hierarchy – use container divs or Tailwind utility classes to position elements similarly (Flexbox/Grid for layout, consistent margins/padding). Ensure the React component tree mirrors the structure of the reference. Keep components small and reusable
medium.com
 (split large components into smaller ones).

6.Apply Styling. Use Tailwind CSS (or your chosen CSS method) to match the exact styles of the original: fonts, font sizes, colors, spacing, borders, etc. Tailwind’s utility classes can enforce pixel-precise spacing and colors, and it’s widely recommended for consistency
medium.com
. If the original uses specific custom fonts or assets, add them to the project (e.g. include Google Fonts or SVG images). Adjust global styles (e.g. a CSS reset or base stylesheet) to match the reference’s base typography and box model. Work section by section to refine styles.

7.Populate Dynamic Data (Optional). If the frontend needs to display dynamic data (e.g. latest signals, account metrics, news), set up data fetching or mock data for now. We have existing code (e.g. fetch_account, fetch_recent_signals) that could eventually be exposed via an API. For the pixel-perfect copy task, static or dummy data is sufficient, but plan for integration with the Python backend later. (E.g., the news list or last signal details can initially be hard-coded to match the reference layout.)

8.Initial Frontend Screenshot. Run the development server (e.g. npm run dev or npm start) and open the new page (e.g. http://localhost:3000). Use MCP Playwright to take a full-page screenshot of the new page (save as page1.png). Ensure the viewport and device emulation match the reference capture (if the original is desktop width, use that).

9.Automated Comparison. Compare original.png vs. page1.png using the MCP Playwright comparison tool. Identify visual differences: layout shifts, font mismatches, color deviations, etc. Tools like Playwright’s expect(page).toHaveScreenshot() can highlight pixel diffs, or simply use an image diffing tool. Note all discrepancies.

10.Fix Visual Discrepancies. Update the frontend code to correct each difference. This may involve tweaking CSS/Tailwind classes, adjusting container sizes, or fixing margins/padding. Refer back to the reference: check exact pixel measurements if needed. After fixes, rebuild and retake a screenshot of the new page.

11.Iterate Until Identical. Repeat the screenshot-compare-fix cycle. Each iteration should reduce differences. Continue until no visible differences remain between the reference and the new page (i.e., the pages are pixel-for-pixel identical). At that point, the frontend matches the design exactly.

12.Implement Interactivity and Responsiveness. Once static layout is perfected, add any interactive features (e.g. form inputs, buttons, hover effects). If the reference site has responsive behaviors (e.g. mobile layouts), implement responsive CSS (Tailwind’s responsive utilities). Test at different screen sizes and repeat visual comparison if needed for those breakpoints.

13.Quality Assurance & Best Practices. Throughout development, follow clean code practices: use clear naming, comment complex logic, and adhere to component-based patterns
medium.com
medium.com
. Lint and format the code automatically (ESLint/Prettier) to ensure consistency
bacancytechnology.com
. Write any necessary unit or integration tests for front-end components. Conduct a code review of the frontend code with peers.

14.Final Verification. Confirm the frontend meets all requirements: it lives in the designated directory (e.g. /frontend), it builds without errors, and the UI matches the original in all sections. Take final screenshots (original_final.png, frontend_final.png) to document the match. Ensure deployment steps or integration with the backend are documented.

Deliverables

Frontend Code: A fully implemented frontend (e.g. in /frontend) with all pages/components matching the reference site.

Screenshots: original.png (reference site) and page1.png (new frontend) showing the page. Final before/after images if needed.

Documentation: Confirmation in README or PR description that the new page matches the reference exactly. List any decisions (chosen framework, directory structure) and mention use of MCP Playwright for visual testing.

Team Roles (for record): Names/roles of team members who implemented the frontend, as formed in Step 1.