/* Resolve the console theme before first paint. Without this the dashboard
   renders dark, then snaps to light a frame later for anyone who picked it —
   worse than either theme on its own. */
const PRE_PAINT = `(function(){var t="dark";try{var v=localStorage.getItem("nidra-console-theme");
if(v==="light"||v==="dark"){t=v;}
else if(v==="system"){t=window.matchMedia("(prefers-color-scheme: light)").matches?"light":"dark";}
}catch(e){}document.documentElement.setAttribute("data-console-theme",t);})();`;

export default function DemoLayout({ children }: LayoutProps<"/demo">) {
  return (
    <>
      <script dangerouslySetInnerHTML={{ __html: PRE_PAINT }} />
      {children}
    </>
  );
}
