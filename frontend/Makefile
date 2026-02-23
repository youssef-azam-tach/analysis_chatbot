include ../.env
export $(shell sed 's/=.*//' ../.env)

.PHONY: clean build publish lint dev

clean:
	@docker image rm --force $(APP_NAME_FRONTEND):$(IMAGE_VERSION) || true

build:
	@docker image rm --force $(APP_NAME_FRONTEND):$(IMAGE_VERSION) || true
	@docker build \
		-t $(APP_NAME_FRONTEND):$(IMAGE_VERSION) .

publish:
	@docker push $(APP_NAME_FRONTEND):$(IMAGE_VERSION)

lint:
	@docker run --rm -i hadolint/hadolint < Dockerfile

dev:
	@npm run dev
