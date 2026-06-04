package com.sihan.chatbot.upload;

public record UploadedFileRecord(
        String file_id,
        String filename,
        String path,
        long size,
        double created_at
) {
}
