// Fetch ChatKit thread state for the Agent panel
export async function fetchThreadState(
  threadId: string,
  portalId: string,
  accountId: string
) {
  const params = new URLSearchParams({
    thread_id: threadId,
    portal_id: portalId,
    account_id: accountId,
  });
  try {
    const res = await fetch(`/chatkit/state?${params.toString()}`);
    if (!res.ok) throw new Error(`State API error: ${res.status}`);
    return res.json();
  } catch (err) {
    console.error("Error fetching thread state:", err);
    return null;
  }
}

export async function fetchBootstrapState(portalId: string, accountId: string) {
  const params = new URLSearchParams({
    portal_id: portalId,
    account_id: accountId,
  });
  try {
    const res = await fetch(`/chatkit/bootstrap?${params.toString()}`);
    if (!res.ok) throw new Error(`Bootstrap API error: ${res.status}`);
    return res.json();
  } catch (err) {
    console.error("Error bootstrapping state:", err);
    return null;
  }
}
