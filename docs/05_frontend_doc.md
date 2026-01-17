Command – Zbuduj nowy frontend dla Trader CRM
Przegląd

Musimy stworzyć samodzielny frontend webowy dla naszego Trader CRM (model EUR/USD), który dokładnie odpowiada projektowi wskazanej strony referencyjnej (link dostarczany osobno). Zastępujemy obecny interfejs Streamlit klasyczną aplikacją webową (np. React/Next.js). Celem jest kopia piksel w piksel: układ, typografia, kolory i komponenty muszą się zgadzać. Zbieramy zespół (frontend developerzy, projektant UI/UX, QA/tester itp.) do realizacji. Użyjemy MCP Playwright do automatycznych porównań zrzutów ekranu, iteracyjnie korygując różnice, aż nowy frontend będzie wizualnie identyczny z oryginałem. Kod frontendu powinien żyć w nowym katalogu (np. `/frontend`), oddzielonym od backendu w Pythonie, aby uniknąć konfliktów i ułatwić nawigację
nairihar.medium.com
netguru.com
. Stosujemy nowoczesne praktyki: framework komponentowy (React z Tailwind CSS), czysty kod (SOLID, sensowne nazwy)
medium.com
, oraz standardy kodu wymuszane przez ESLint/Prettier
bacancytechnology.com
.

Kroki

1. Zbierz zespół wdrożeniowy. Zidentyfikuj i przypisz role potrzebne do projektu frontendu (np. jeden lub kilku developerów React/Next.js lub Vue, projektant UI/UX do layoutów, QA/tester do testów wizualnych, specjalista DevOps/CI). Zdefiniuj odpowiedzialności i kanały komunikacji. Zespół cross‑funkcyjny zapewnia spójność designu, developmentu i testów.

2. Wybierz technologię frontendu. Dobierz framework webowy do naszych potrzeb. Popularne opcje to React (często z Next.js), Vue lub Angular. Dla tego projektu polecany jest React (z TypeScript) ze względu na architekturę komponentową i ekosystem. Użyjemy frameworka CSS utility-first jak Tailwind CSS lub podejścia z modułami CSS, by osiągnąć precyzyjne style. (Nacisk Reacta na wielokrotne użycie komponentów i hooki dobrze pasuje do zakresu
medium.com
.) Wymuszaj rygorystyczne standardy: TypeScript w trybie strict oraz lintowanie (ESLint) i formatowanie (Prettier) dla spójności
bacancytechnology.com
.

3. Ustaw strukturę projektu. Utwórz w repozytorium nowy katalog najwyższego poziomu (np. `frontend/` lub `ui/`) dla aplikacji webowej
nairihar.medium.com
. Zainicjalizuj projekt (np. create-react-app, starter Next.js, Vite) wewnątrz tego folderu. Zorganizuj go wg dobrych praktyk: katalogi components/, pages/ lub views/, hooks/, assets/, services/
netguru.com
. Taki podział izoluje backend i frontend. Skonfiguruj importy absolutne lub aliasy ścieżek (przez jsconfig.json lub tsconfig.json), aby uprościć importy w komponentach
netguru.com
. Zacommituj początkową strukturę do repozytorium.

4. Uchwyć projekt referencyjny. Otwórz dostarczony link w przeglądarce. Użyj narzędzia MCP Playwright do zrobienia pełnego zrzutu ekranu (np. mcp playwright lub równoważny skrypt). Zapisz go jako `original.png`. To nasza baza. Obejrzyj strukturę strony — zidentyfikuj sekcje (nagłówki, nawigacja, bloki treści, stopki). Zanotuj fonty, kolory i odstępy w razie potrzeby. Ta referencja prowadzi implementację.

5. Zbuduj wstępny layout strony. W nowym projekcie frontendu utwórz pierwszą stronę (np. `pages/index.tsx` dla Next.js lub App.js dla CRA). Odwzoruj strukturę referencji sekcja po sekcji: twórz komponenty React dla głównych bloków (header, hero/baner, features, wykresy, footer itd.). Użyj placeholderów lub skopiuj tekst z referencji. Na tym etapie skup się na układzie i hierarchii — stosuj kontenery lub klasy Tailwind do ułożenia elementów (Flexbox/Grid, spójne marginesy/padding). Upewnij się, że drzewo komponentów odzwierciedla strukturę referencji. Trzymaj komponenty małe i wielorazowe
medium.com
 (duże dziel na mniejsze).

6. Zastosuj stylowanie. Użyj Tailwind CSS (lub wybranej metody CSS), aby dopasować style oryginału: fonty, rozmiary, kolory, odstępy, ramki itp. Klasy utility Tailwind zapewniają pikselową precyzję i spójność
medium.com
. Jeśli oryginał korzysta ze specyficznych fontów lub assetów, dodaj je do projektu (np. Google Fonts, SVG). Dostosuj style globalne (reset CSS lub bazowy stylesheet), aby pasowały do bazowej typografii i box modelu referencji. Pracuj sekcja po sekcji, doprecyzowując style.

7. Wypełnij danymi dynamicznymi (opcjonalnie). Jeśli frontend ma pokazywać dane dynamiczne (np. ostatnie sygnały, metryki konta, newsy), skonfiguruj pobieranie danych lub mocki. Mamy istniejący kod (np. fetch_account, fetch_recent_signals), który można wystawić przez API. Do kopii piksel w piksel wystarczą statyczne/przykładowe dane, ale zaplanuj późniejszą integrację z backendem Pythona. (Np. lista newsów czy szczegóły ostatniego sygnału mogą być początkowo zahardkodowane, by pasowały do layoutu referencyjnego.)

8. Pierwszy zrzut frontendu. Uruchom serwer deweloperski (np. `npm run dev` lub `npm start`) i otwórz nową stronę (np. http://localhost:3000). Użyj MCP Playwright, aby zrobić pełny zrzut ekranu nowej strony (zapisz jako `page1.png`). Upewnij się, że viewport/emulacja urządzenia odpowiada ujęciu referencyjnemu (jeśli oryginał to desktop, użyj tej szerokości).

9. Automatyczne porównanie. Porównaj `original.png` z `page1.png` narzędziem MCP Playwright do porównań. Zanotuj różnice wizualne: przesunięcia układu, niezgodne fonty, kolory itd. Narzędzia jak Playwright `expect(page).toHaveScreenshot()` podświetlają różnice pikselowe; można też użyć prostego image diff. Spisz wszystkie rozbieżności.

10. Napraw różnice wizualne. Zaktualizuj kod frontendu, by skorygować każdą różnicę. Może to wymagać korekty klas CSS/Tailwind, rozmiarów kontenerów lub marginesów/paddingu. Odnoś się do referencji: sprawdzaj dokładne wymiary pikseli, jeśli trzeba. Po poprawkach zbuduj i ponownie wykonaj zrzut nowej strony.

11. Iteruj do identyczności. Powtarzaj cykl zrzut → porównanie → poprawki. Każda iteracja powinna zmniejszać różnice. Kontynuuj, aż nie będzie widocznych rozbieżności między referencją a nową stroną (piksel w piksel). W tym momencie frontend odpowiada projektowi.

12. Dodaj interakcje i responsywność. Gdy statyczny layout jest dopracowany, dodaj funkcje interaktywne (np. pola formularzy, przyciski, efekty hover). Jeśli referencja ma responsywność (np. mobile), zaimplementuj responsywne CSS (utility Tailwind). Testuj różne szerokości i powtórz porównania dla tych breakpointów.

13. Zapewnienie jakości i dobre praktyki. W trakcie developmentu stosuj zasady czystego kodu: jasne nazwy, komentarze przy złożonej logice, wzorce komponentowe
medium.com
medium.com
. Automatycznie lintuj i formatuj kod (ESLint/Prettier) dla spójności
bacancytechnology.com
. Dodaj potrzebne testy jednostkowe/integracyjne komponentów frontendu. Przeprowadź code review z zespołem.

14. Weryfikacja końcowa. Potwierdź, że frontend spełnia wymagania: żyje w wyznaczonym katalogu (np. `/frontend`), buduje się bez błędów, a UI odpowiada oryginałowi w każdej sekcji. Zrób finalne zrzuty (original_final.png, frontend_final.png) jako dowód zgodności. Upewnij się, że kroki wdrożenia lub integracji z backendem są udokumentowane.

Rezultaty

Kod frontendu: w pełni zaimplementowany frontend (np. w `/frontend`) z wszystkimi stronami/komponentami odpowiadającymi stronie referencyjnej.

Zrzuty ekranu: `original.png` (strona referencyjna) i `page1.png` (nowy frontend) pokazujące stronę. W razie potrzeby finalne obrazy przed/po.

Dokumentacja: potwierdzenie w README lub opisie PR, że nowa strona dokładnie pasuje do referencji. Lista decyzji (wybrany framework, struktura katalogów) oraz wzmianka o użyciu MCP Playwright do testów wizualnych.

Role zespołu (dla ewidencji): imiona/role członków zespołu, którzy zbudowali frontend, zgodnie z krokiem 1.
