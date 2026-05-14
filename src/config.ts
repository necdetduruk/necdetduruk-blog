export const SITE = {
  website: "https://necdetduruk.dev",  // or .com — placeholder for now
  author: "Necdet Duruk",
  profile: "https://www.linkedin.com/in/necdetduruk/",
  desc: "Senior ML Engineer writing about time-series, foundation models, and production ML.",
  title: "Necdet Duruk",
  ogImage: "astropaper-og.jpg",
  lightAndDarkMode: true,
  postPerIndex: 4,
  postPerPage: 4,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showBackButton: true, // show back button in post detail
  editPost: {
  enabled: false,    // simplest — just turn off the "Edit page" link
  text: "Edit page",
  url: "https://github.com/satnaing/astro-paper/edit/main/",
},
  dynamicOgImage: true,
  dir: "ltr", // "rtl" | "auto"
  lang: "en", // html lang code. Set this empty and default will be "en"
  timezone: "America/Toronto", // Default global timezone (IANA format) https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
} as const;
