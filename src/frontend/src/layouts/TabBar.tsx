import { NavLink } from "react-router";
import { ROUTES } from "../routes";

interface TabBarProps {
  sessionId: string;
}

/** Links to a session's tabs, with the current one highlighted. */
export function TabBar({ sessionId }: TabBarProps) {
  const tabs = [
    { to: ROUTES.input(sessionId), label: "Input" },
    { to: ROUTES.acReview(sessionId), label: "AC Review" },
    { to: ROUTES.uatReview(sessionId), label: "UAT Review" },
    { to: ROUTES.export(sessionId), label: "Export" },
  ];

  return (
    <nav className="flex shrink-0 gap-1 border-b border-gray-200 px-6 py-3">
      {tabs.map((tab) => (
        <NavLink
          key={tab.to}
          to={tab.to}
          className={({ isActive }) =>
            `rounded-lg px-4 py-2 text-sm font-semibold uppercase tracking-wide ${
              isActive ? "bg-primary text-white" : "text-gray-700 hover:bg-gray-100"
            }`
          }
        >
          {tab.label}
        </NavLink>
      ))}
    </nav>
  );
}
