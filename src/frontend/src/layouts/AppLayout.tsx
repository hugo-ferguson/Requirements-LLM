import { Outlet } from "react-router";
import { SessionSidebar } from "./SessionSidebar";

/** The page frame: the session sidebar beside the current page. */
export function AppLayout() {
  return (
    <div className="flex h-screen bg-white">
      <SessionSidebar />
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  );
}
