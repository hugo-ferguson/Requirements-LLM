import { useNavigate } from "react-router";
import { useSessions } from "../context/SessionsContext";
import { ROUTES } from "../routes";

export function EmptyStatePage() {
  const navigate = useNavigate();
  const { createSession } = useSessions();

  async function handleCreateSession() {
    const created = await createSession();
    navigate(ROUTES.input(created.id));
  }

  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 text-gray-500">
      <h2 className="text-xl font-semibold text-gray-700">No session selected</h2>
      <p>Create a new session to start generating acceptance criteria.</p>
      <button
        type="button"
        onClick={handleCreateSession}
        className="rounded-full bg-primary px-6 py-2 font-medium text-white hover:bg-primary-hover"
      >
        Create new session
      </button>
    </div>
  );
}
