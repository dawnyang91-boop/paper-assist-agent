package com.sihan.chatbot.upload;

import com.sihan.chatbot.common.SessionUser;
import com.sihan.chatbot.config.ChatbotProperties;
import com.sihan.chatbot.lock.RedisLockService;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

@Service
public class UploadService {
    private static final String FILE_SET_PREFIX = "upload:files:";
    private static final String FILE_META_PREFIX = "upload:meta:";
    private static final String FILE_HASH_PREFIX = "upload:file:";

    private final ChatbotProperties properties;
    private final StringRedisTemplate redisTemplate;
    private final RedisLockService lockService;

    public UploadService(
            ChatbotProperties properties,
            StringRedisTemplate redisTemplate,
            RedisLockService lockService
    ) {
        this.properties = properties;
        this.redisTemplate = redisTemplate;
        this.lockService = lockService;
    }

    public List<UploadedFileRecord> replaceFiles(List<MultipartFile> files, SessionUser user) {
        if (files == null || files.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "请选择要上传的文件");
        }
        clearFiles(user);
        List<UploadedFileRecord> saved = new ArrayList<>();
        for (MultipartFile file : files) {
            saved.add(saveOne(file, user));
        }
        return saved;
    }

    public List<UploadedFileRecord> listFiles(SessionUser user) {
        List<UploadedFileRecord> records = new ArrayList<>();
        var fileIds = redisTemplate.opsForSet().members(fileSetKey(user));
        if (fileIds == null) {
            return records;
        }
        for (String fileId : fileIds) {
            UploadedFileRecord record = readRecord(user, fileId);
            if (record != null) {
                records.add(record);
            }
        }
        records.sort(Comparator.comparingDouble(UploadedFileRecord::created_at).reversed());
        return records;
    }

    public boolean deleteFile(String fileId, SessionUser user) {
        UploadedFileRecord record = readRecord(user, fileId);
        if (record == null) {
            return false;
        }
        try {
            Files.deleteIfExists(Path.of(record.path()));
        } catch (IOException ignored) {
            // Metadata is still removed so the pending batch remains consistent.
        }
        redisTemplate.opsForSet().remove(fileSetKey(user), fileId);
        redisTemplate.delete(metaKey(user, fileId));
        return true;
    }

    public String uploadDir(SessionUser user) {
        return uploadRoot().resolve(safeUserId(user.id())).toAbsolutePath().normalize().toString();
    }

    public void clearFiles(SessionUser user) {
        for (UploadedFileRecord record : listFiles(user)) {
            deleteFile(record.file_id(), user);
        }
        redisTemplate.delete(fileSetKey(user));
    }

    private UploadedFileRecord saveOne(MultipartFile file, SessionUser user) {
        validate(file);
        String hash = sha256(file);
        String hashKey = FILE_HASH_PREFIX + safeUserId(user.id()) + ":" + hash;
        RedisLockService.LockHandle lock = lockService.tryLock("lock:upload:" + safeUserId(user.id()) + ":" + hash, Duration.ofSeconds(30));
        if (lock == null) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "相同文件正在上传，请稍后再试");
        }
        try {
            String existingId = redisTemplate.opsForValue().get(hashKey);
            if (StringUtils.hasText(existingId)) {
                UploadedFileRecord existing = readRecord(user, existingId);
                if (existing != null && Files.exists(Path.of(existing.path()))) {
                    redisTemplate.opsForSet().add(fileSetKey(user), existing.file_id());
                    return existing;
                }
            }

            String originalFilename = cleanFilename(file.getOriginalFilename());
            String extension = extensionOf(originalFilename);
            String fileId = UUID.randomUUID().toString();
            Path userDir = uploadRoot().resolve(safeUserId(user.id())).toAbsolutePath().normalize();
            Files.createDirectories(userDir);
            Path target = userDir.resolve(fileId + extension);
            file.transferTo(target);

            UploadedFileRecord record = new UploadedFileRecord(
                    fileId,
                    originalFilename,
                    target.toString(),
                    file.getSize(),
                    System.currentTimeMillis() / 1000.0
            );
            writeRecord(user, record);
            redisTemplate.opsForSet().add(fileSetKey(user), fileId);
            redisTemplate.opsForValue().set(hashKey, fileId, Duration.ofDays(30));
            return record;
        } catch (IOException exc) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "文件保存失败：" + exc.getMessage(), exc);
        } finally {
            lockService.unlock(lock);
        }
    }

    private void validate(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "上传文件为空");
        }
        long maxBytes = properties.getMaxFileSizeMb() * 1024L * 1024L;
        if (file.getSize() > maxBytes) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "文件过大");
        }
        String filename = cleanFilename(file.getOriginalFilename());
        String extension = extensionOf(filename).replace(".", "").toLowerCase();
        if (!properties.getAllowedExtensions().contains(extension)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "不支持的文件类型：" + extension);
        }
    }

    private String sha256(MultipartFile file) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (InputStream input = file.getInputStream(); DigestInputStream stream = new DigestInputStream(input, digest)) {
                byte[] buffer = new byte[8192];
                while (stream.read(buffer) != -1) {
                    // DigestInputStream updates the digest as bytes are read.
                }
            }
            return HexFormat.of().formatHex(digest.digest());
        } catch (IOException | NoSuchAlgorithmException exc) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "计算文件指纹失败", exc);
        }
    }

    private void writeRecord(SessionUser user, UploadedFileRecord record) {
        redisTemplate.opsForHash().putAll(metaKey(user, record.file_id()), Map.of(
                "file_id", record.file_id(),
                "filename", record.filename(),
                "path", record.path(),
                "size", String.valueOf(record.size()),
                "created_at", String.valueOf(record.created_at())
        ));
    }

    private UploadedFileRecord readRecord(SessionUser user, String fileId) {
        Map<Object, Object> fields = redisTemplate.opsForHash().entries(metaKey(user, fileId));
        if (fields.isEmpty()) {
            return null;
        }
        return new UploadedFileRecord(
                String.valueOf(fields.get("file_id")),
                String.valueOf(fields.get("filename")),
                String.valueOf(fields.get("path")),
                Long.parseLong(String.valueOf(fields.get("size"))),
                Double.parseDouble(String.valueOf(fields.get("created_at")))
        );
    }

    private Path uploadRoot() {
        return Path.of(properties.getUploadRoot());
    }

    private String fileSetKey(SessionUser user) {
        return FILE_SET_PREFIX + safeUserId(user.id());
    }

    private String metaKey(SessionUser user, String fileId) {
        return FILE_META_PREFIX + safeUserId(user.id()) + ":" + fileId;
    }

    private String safeUserId(String userId) {
        return userId == null ? "local" : userId.replaceAll("[^a-zA-Z0-9_.@-]", "_");
    }

    private String cleanFilename(String filename) {
        String value = StringUtils.hasText(filename) ? filename : "uploaded.txt";
        return Path.of(value).getFileName().toString().replaceAll("[\\\\/:*?\"<>|]", "_");
    }

    private String extensionOf(String filename) {
        int index = filename.lastIndexOf('.');
        if (index < 0) {
            return ".txt";
        }
        return filename.substring(index).toLowerCase();
    }
}
