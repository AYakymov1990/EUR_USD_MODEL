Command – Zaimplementuj stronę lądowania CRM
Przegląd

Stwórz pierwszą (landing) stronę aplikacji webowej Trader CRM inspirowaną sekcją hero Twenty.com, ale skup się na funkcjonalności zamiast perfekcyjnego dopasowania pikseli. Strona powinna wyraźnie pokazywać informacje o koncie użytkownika, najnowszy sygnał transakcyjny i dwa przyciski „LONG” oraz „SHORT” do potwierdzania transakcji. Zgodnie z dokumentacją projektową frontend to Next.js 13 (App Router) z Tailwind CSS, więc buduj całą nową UI w tym frameworku. W szczególności edytuj `src/app/page.tsx` (oraz `globals.css`), aby zaimplementować tę stronę. Ogólne wrażenia (fonty, kolory, odstępy) powinny w przybliżeniu nawiązywać do referencji Twenty.com (motyw „Open-Source CRM”), ale perfekcyjna zgodność pikseli nie jest wymagana; priorytetem jest funkcjonalność i czysty, responsywny layout. Użyj MCP Playwright do automatycznych zrzutów i porównań!

Kroki

Analizuj istniejący kod i stack:

Potwierdź z README, że frontend używa Next.js 13 (App Router) i Tailwind CSS. Punkt wejścia landing page to `frontend/src/app/page.tsx`, a style globalne są w `src/app/globals.css`.

W dokumentacji jest zapis, że nowy UI ma nawiązywać do sekcji hero Twenty.com (motyw „#1 Open-Source CRM”). Użyj tego jako przewodnika layoutu i brandingu (np. nagłówek „Trader CRM” lub „Open-Source Trader CRM”, aby wskazać cel aplikacji).

Skonfiguruj środowisko developerskie:

Uruchom frontend lokalnie:

cd frontend 
npm install
npm run dev


(Uruchamia aplikację Next.js na http://localhost:3000.)

Uruchom też backend API (FastAPI) zgodnie z głównym README, aby endpointy konta i sygnałów były dostępne. Frontend używa zmiennej `NEXT_PUBLIC_API_BASE` do wskazania bazowego URL API (domyślnie http://localhost:8000). Upewnij się, że jest ustawiona poprawnie (np. w `.env.local`), aby strona mogła pobierać realne dane.

Zbuduj layout i sekcję hero:

W `src/app/page.tsx` zbuduj ogólną strukturę. Naśladuj hero z Twenty.com: wyraźny nagłówek, podtytuł i strefę call-to-action. Na przykład użyj dużego tytułu typu „Trader CRM” z opisem (np. „Zarządzaj sygnałami i zleceniami”), opcjonalnie tła lub grafiki.

Poniżej hero umieść osobne sekcje/karty dla informacji o koncie i sygnale. Używaj semantycznego HTML (`<section>`, `<header>` itp.) i utility Tailwind do układu (np. flex lub grid). Zachowaj czysty design — stosuj spacing i typografię Tailwind, by dopasować się do referencji.

Dodaj widoczny sygnał, że to CRM/dashboard. Na przykład mały tagline lub ikona z napisem „Trader CRM” albo „Trading Dashboard”, aby użytkownik od razu rozumiał cel aplikacji.

Wyświetl informacje o koncie:

Dodaj elementy UI (np. karty/panele) prezentujące dane konta. W kodzie React (po stronie klienta) pobierz dane konta z `GET /account` backendu
GitHub
. Możesz to zrobić w hooku `useEffect` lub Next.js data fetching.

Renderuj kluczowe pola konta na stronie: np. saldo, ID konta i tryb demo/live. Formatuj je czytelnie z etykietami („Saldo: $10,000”). Obsłuż stany ładowania i błędu (np. pokazuj „Loading…” lub „Error loading account”).

Wyświetl sygnał transakcyjny:

Dodaj sekcję dla najnowszego sygnału. Użyj endpointu `POST /signals/generate` (aby pobrać nową predykcję) lub `GET /signals/recent` (aby pokazać ostatni sygnał)
GitHub
. Pobierz sygnał przy załadowaniu strony (lub na żądanie) i wyświetl wynik.

Pokaż co najmniej przewidywaną akcję (Long lub Short) i ewentualne dodatkowe info (np. confidence lub krótkie uzasadnienie). Przykład: „Sygnał: Long (prognozowany wzrost ceny)”. Opcjonalnie dodaj timestamp lub możliwość wygenerowania sygnału przyciskiem.

Zaimplementuj przyciski LONG/SHORT:

Umieść dwa przyciski LONG i SHORT w pobliżu sekcji sygnału. Odróżnij je stylistycznie (np. zielony/czerwony) klasami Tailwind.

W handlerach kliknięć wywołuj `POST /orders/market` na backendzie z `{ "action": "long" }` lub `{ "action": "short" }`
GitHub
. Użyj np. fetch lub Axios do wysłania żądania.

Podczas trwania żądania zablokuj przyciski lub pokaż spinner. Po odpowiedzi pokaż feedback: np. komunikat „Zlecenie wysłane” albo zaktualizuj saldo, jeśli dostępne. Obsłuż błędy, wyświetlając alert lub wiadomość.

Styl i responsywność:

Używaj Tailwind CSS do całego stylowania. Stosuj klasy utility dla layoutu (np. flex, grid, p-4), typografii (np. text-2xl, font-bold) i kolorów zgodnych z paletą Twenty.com (np. odcienie niebieskiego i szarości). Możesz edytować `src/app/globals.css` dla globalnych fontów/kolorów, jeśli potrzeba
GitHub
.

Zapewnij responsywność: używaj prefixów responsywnych (md:w-1/2, sm:text-lg itp.), aby wyglądało dobrze na mobile i desktopie. Sprawdź layout przy różnych szerokościach i dostosuj spacing.

Testuj i iteruj:

Odśwież http://localhost:3000 i sprawdź, czy wszystkie sekcje renderują się poprawnie. Dane konta powinny się pojawić, sygnał powinien się ładować, a przyciski LONG/SHORT powinny działać end-to-end.

Porównaj layout z referencją. Rozmieszczenie nagłówków, przycisków i tekstu powinno strukturalnie nawiązywać do hero Twenty.com (np. nagłówek na środku, bloki treści w kolejności), ale drobne odchylenia są OK. Nie spędzaj czasu na perfekcyjnym wyrównaniu — ważniejsze są czytelność i pełna funkcjonalność.

Napraw ewentualne problemy wizualne lub funkcjonalne: dostosuj marginesy, padding lub rozmiary komponentów, aby sekcje były proporcjonalne. Powtarzaj testy (w tym przypadki błędów API), aż wszystko będzie działać płynnie.

Rezultaty

Zaktualizowany kod w folderze frontendu: nowa strona lądowania w `src/app/page.tsx` (i ewentualne nowe komponenty) ze stylowaniem Tailwind.

Sprawna integracja: strona powinna pobierać z `/account` i `/signals`, wyświetlać dane oraz wysyłać zlecenia przez `/orders/market`
GitHub
.

Spójność designu: layout i styl nawiązują do motywu Twenty.com (CRM) zgodnie z opisem, wykorzystując obecny setup Tailwind
GitHub
.

Weryfikacja: krótkie potwierdzenie (lub zrzuty), że na http://localhost:3000 widać informacje o koncie, sygnał oraz działające przyciski LONG/SHORT. (Może to być prosta informacja, że strona działa i odzwierciedla kontekst CRM.)
