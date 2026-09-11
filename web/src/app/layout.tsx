import type { Metadata } from "next";
import { Crimson_Pro, DM_Sans, Poppins } from "next/font/google";
import "./globals.css";

const dmSans = DM_Sans({
  variable: "--font-dm-sans",
  subsets: ["latin"],
  weight: ["400", "500", "700"],
});

const poppins = Poppins({
  variable: "--font-poppins",
  subsets: ["latin"],
  weight: ["400", "600", "700"],
});

const crimsonPro = Crimson_Pro({
  variable: "--font-crimson-pro",
  subsets: ["latin"],
  weight: ["400", "600"],
  style: ["italic", "normal"],
});

export const metadata: Metadata = {
  title: "NIDRA — Predictive Network World Model",
  description:
    "NIDRA learns the dynamics of network traffic and rolls them forward to forecast host compromise minutes before it lands — with a confidence band and a lead time. Not an intrusion detector.",
  icons: { icon: "/seo/favicon.png" },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      /* /demo sets data-console-theme on this element before paint, so the
         server markup and the hydrated DOM differ here by design. */
      suppressHydrationWarning
      className={`${dmSans.variable} ${poppins.variable} ${crimsonPro.variable} h-full scroll-smooth antialiased`}
    >
      <body className="font-body text-gray-dark selection:bg-primary-blue selection:text-white min-h-full flex flex-col bg-white">
        {children}
      </body>
    </html>
  );
}
