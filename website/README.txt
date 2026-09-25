ApnaAlgo website — Hostinger upload instructions
=================================================

WHAT'S IN HERE
--------------
index.html   The entire site: home page, Login/Sign up, and Blog.
             One file, no build step, no dependencies to install.

HOW TO UPLOAD (Hostinger File Manager)
---------------------------------------
1. Log in to hPanel (hostinger.com).
2. Go to Websites > apnalgo.ai/apnaalgo.ai > Manage.
3. Open File Manager.
4. Go into the public_html folder (this is the web root).
5. If there's an existing index.html (or any old site files) in there,
   move them into a backup folder first — don't delete anything until
   you've confirmed the new site works.
6. Upload index.html from this folder into public_html.
7. Visit your domain in a browser. It should load immediately —
   no extra configuration needed.

ALTERNATIVE: UPLOAD VIA FTP
----------------------------
If you prefer FTP (FileZilla, etc.):
  Host: your Hostinger FTP hostname (in hPanel > Advanced > FTP Accounts)
  Upload index.html to /public_html/

HOW THE SITE IS STRUCTURED
---------------------------
It's a single HTML file with inline CSS and JavaScript — everything
needed to run is in that one file except two things it loads from the
internet:
  - Google Fonts (Sora, IBM Plex Sans, IBM Plex Mono)
  - Nothing else. No frameworks, no build tools, no npm.

Navigation between "pages" (Home, Login/Sign up, Blog, individual blog
posts) works through the URL's hash (#login, #blog, #blog-<slug>) using
JavaScript — this means it works on any static host, including
Hostinger's shared hosting, with zero server configuration.

WHAT STILL NEEDS YOUR INPUT
-----------------------------
1. LOGO — the header currently uses a placeholder green "अ" mark.
   Send the real logo/wordmark files and they'll be swapped in.
2. BLOG POSTS — the 4 posts are realistic sample content in a Hinglish
   tone (matching the app), meant as a starting structure. Replace or
   approve before this goes live publicly.
3. LOGIN / SIGN UP — the forms are front-end only right now. They show
   a "preview only" message on submit instead of creating a real
   account. This needs a real backend (custom API, Firebase, Supabase,
   etc.) before it can go live — tell your developer which one and the
   JavaScript submit handlers (near the bottom of index.html, search
   for "previewSubmit") are the place to wire in real requests.
4. PRICING — currently shows "Request a Quote" with no fixed number,
   per your instruction. Update the request-handling flow (currently
   just a front-end confirmation message) to actually notify you by
   email once you're ready to receive quote requests for real — this
   also needs a backend or a form service (e.g. Formspree) wired into
   the #requestForm submit handler.

QUESTIONS
---------
Reply in your Claude Code session and changes can be made directly to
index.html, then this package can be regenerated.
