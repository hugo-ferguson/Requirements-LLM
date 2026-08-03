import { request } from "./client";
import type { paths } from "./schema";
import type { ChatMessage } from "../types/conversation";

export type ConversationRequestBody =
  paths["/conversation/messages"]["post"]["requestBody"]["content"]["application/json"];
type ConversationMessageWire = ConversationRequestBody["messages"][number];
export type ChatReply =
  paths["/conversation/messages"]["post"]["responses"]["200"]["content"]["application/json"];
export type GenerateResult =
  paths["/conversation/generate"]["post"]["responses"]["200"]["content"]["application/json"];

function toWireMessage(message: ChatMessage): ConversationMessageWire {
  return { role: message.role, text: message.text, attachments: message.attachments };
}

export const conversationApi = {
  sendMessage: (messages: ChatMessage[]): Promise<ChatReply> =>
    request<ChatReply>("/conversation/messages", {
      method: "POST",
      body: JSON.stringify({ messages: messages.map(toWireMessage) } satisfies ConversationRequestBody),
    }),
  generate: (messages: ChatMessage[]): Promise<GenerateResult> =>
    request<GenerateResult>("/conversation/generate", {
      method: "POST",
      body: JSON.stringify({ messages: messages.map(toWireMessage) } satisfies ConversationRequestBody),
    }),
};
