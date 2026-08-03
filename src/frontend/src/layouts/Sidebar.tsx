import { NavLink } from "react-router";
import { ROUTES } from "../routes";

const NAV_ITEMS = [
  { to: ROUTES.input, label: "Input" },
  { to: ROUTES.acReview, label: "AC Review" },
  { to: ROUTES.uatReview, label: "UAT Review" },
  { to: ROUTES.export, label: "Export" },
];

export function Sidebar() {
  return (
    <aside className="flex w-56 shrink-0 flex-col gap-6 border-r border-gray-200 p-6">
      <h1 className="text-lg font-bold tracking-wide text-gray-900">STORY2SPEC</h1>
      <nav className="flex flex-col gap-1">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `rounded-lg px-4 py-2 text-sm font-semibold uppercase tracking-wide ${
                isActive ? "bg-primary text-white" : "text-gray-700 hover:bg-gray-100"
              }`
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
